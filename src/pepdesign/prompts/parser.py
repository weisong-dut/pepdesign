"""从模型输出中抽取多肽序列。

采用多级回退：强标记、XML 标签、代码块、逐行扫描、全文兜底。
每级归一化时剥离末端修饰与 PEG、转换三字母码，并把修饰信息交给奖励函数。
解析失败返回分类原因，供调用方区分类别并统计失败率。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pepdesign.bio.aa import ALLOWED_AA, STANDARD_AA, THREE_TO_ONE
from pepdesign.bio.sequence import (
    MAX_PEPTIDE_LEN,
    MIN_PEPTIDE_LEN,
    has_noncanonical_residues,
    strip_terminal_modifications,
)
from pepdesign.prompts.templates import SEQ_OPEN

# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------
_THREE_LETTER_ALT = "|".join(sorted(THREE_TO_ONE, key=len, reverse=True))
# 三字母码：用于把 "Lys-Arg-Trp" 转成 "KRW"
_THREE_LETTER_RE = re.compile(r"\b(" + _THREE_LETTER_ALT + r")\b", re.IGNORECASE)
# 两个及以上三字母码被连字符相连，是多肽三字母写法的独有特征。
_THREE_LETTER_HYPHEN_RE = re.compile(
    r"\b(?:" + _THREE_LETTER_ALT + r")\s*[-－]\s*(?:" + _THREE_LETTER_ALT + r")\b",
    re.IGNORECASE,
)
# PEG 修饰。它的字母 P/E/G 全是合法氨基酸码，不先屏蔽会造成插入突变。
_PEG_RE = re.compile(r"\bPEG\s*\d*\b", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)
_XML_TAG_RE = re.compile(r"<(?:answer|seq|sequence|output)>(.*?)</(?:answer|seq|sequence|output)>",
                         re.IGNORECASE | re.DOTALL)
# "序列: KWKLFKK" 这类前缀会拉低氨基酸占比判据。
# 长度上限 20 个字符，结尾必须是 `:` 或 `=`，序列本身不含这些字符。
_SEQ_PREFIX_RE = re.compile(
    r"^\s*[\w\s]{0,20}?(?:序列|sequence|seq|peptide|answer|output|result)\s*[:：=]\s*",
    re.IGNORECASE,
)

# 一行"看起来像序列"的判定：氨基酸字母占非空白字符的比例
_SEQ_LIKE_MIN_RATIO = 0.8
# 三字母码转换的门槛：命中总长度占非空白字符的比例
_MIN_THREE_LETTER_COVERAGE = 0.5
# 逐行扫描时，短于这个长度的行一律不认（见 _looks_like_sequence 的说明）
_SEQ_LIKE_MIN_LEN = 5
# 全文兜底策略的准入门槛：氨基酸字母占非空白字符的比例
_WHOLE_TEXT_MIN_RATIO = 0.9
# 命中三字母码写法时，转为"宽松模式"的逐行占比门槛
_RELAXED_RATIO = 0.6


@dataclass
class ParsedSequence:
    """解析结果；`ok=False` 时 `sequence` 为空，`reason` 说明失败原因。"""

    ok: bool
    sequence: str = ""
    reason: str = "ok"
    strategy: str = ""                     # 命中的回退层级，用于统计各层贡献
    modifications: list[str] = field(default_factory=list)
    trailing_text: str = ""                # 序列之后的文本（设计理由等）

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "sequence": self.sequence,
            "reason": self.reason,
            "strategy": self.strategy,
            "modifications": self.modifications,
            "length": len(self.sequence),
        }


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def _should_convert_three_letter(s: str) -> bool:
    """判断文本是三字母码写法，而非恰好含氨基酸单词的普通文本。

    满足其一即认定：存在三字母码连字符模式，或三字母码覆盖非空白字符 ≥ 50%
    且至少命中两个码。单个 `HIS` 只会命中一个码，不会被误转换。
    """
    if _THREE_LETTER_HYPHEN_RE.search(s):
        return True
    matches = list(_THREE_LETTER_RE.finditer(s))
    if len(matches) < 2:
        return False
    matched = sum(m.end() - m.start() for m in matches)
    non_space = sum(1 for ch in s if not ch.isspace())
    if non_space == 0:
        return False
    return matched / non_space >= _MIN_THREE_LETTER_COVERAGE


def normalize_candidate(text: str) -> tuple[str, list[str]]:
    """把候选文本归一化成纯氨基酸序列，并返回识别到的修饰。

    处理顺序：剥离末端修饰 → 屏蔽 PEG → 三字母码转单字母 → 去非氨基酸字符。
    """
    mods: list[str] = []
    s = text.strip()

    # --- 1. 末端修饰（与 data/schema.py 共用 bio/ 层的同一套规则） ---
    s, term_mods = strip_terminal_modifications(s)
    mods.extend(term_mods)

    # --- 2. PEG 屏蔽 ---
    if _PEG_RE.search(s):
        mods.append("pegylated")
        s = _PEG_RE.sub(" ", s)

    # --- 3. 三字母码（准入判据见 _should_convert_three_letter） ---
    three_letter = _should_convert_three_letter(s)
    if three_letter:
        s = _THREE_LETTER_RE.sub(lambda mo: THREE_TO_ONE.get(mo.group(0).upper(), ""), s)

    # --- 3.5 非标准残基检测（须在大写化之前、三字母码转换之后） ---
    # 小写字母表示 D-型残基，U/O 表示硒代半胱氨酸/吡咯赖氨酸，会被后续过滤抹掉。
    if not three_letter and has_noncanonical_residues(s):
        mods.append("noncanonical_residues")

    # --- 4. 清掉所有非氨基酸字符（空格、连字符、标点、数字） ---
    s = "".join(ch for ch in s.upper() if ch in ALLOWED_AA)
    return s, mods


def _sequence_body(text: str) -> str:
    """剥掉末端修饰与 PEG 后的文本，用作"字母占比"的分母。

    必须与 `normalize_candidate` 使用同一套剥离顺序，否则分子与分母口径不一致。
    """
    body, _ = strip_terminal_modifications(text)
    return _PEG_RE.sub("", body)


def _ascii_letter_count(s: str) -> int:
    """统计 ASCII 字母个数；`str.isalpha()` 会把汉字也算进去。"""
    return sum(1 for ch in s if ch.isascii() and ch.isalpha())


def _case_is_plausible(s: str) -> bool:
    """判断大小写形态是否像序列而不是英文句子。

    序列统一大写或全小写，英文句子是句首大写的混排。先剥末端修饰
    （`Ac-` 中的 `c` 不算小写），三字母码写法单独放行。
    """
    body, _ = strip_terminal_modifications(s)
    letters = [ch for ch in body if ch.isascii() and ch.isalpha()]
    if not letters:
        return False
    if all(ch.isupper() for ch in letters) or all(ch.islower() for ch in letters):
        return True
    return _should_convert_three_letter(s)


def _word_shape_is_plausible(s: str) -> bool:
    """判断词形是否像序列：最多两个词，或是三字母码写法。

    用于拦住全大写散文：它在大小写形态与氨基酸占比上都与序列相似，只差空格。
    """
    return len(s.split()) <= 2 or _should_convert_three_letter(s)


def _looks_like_sequence(line: str, *, min_len: int, max_len: int) -> bool:
    """判断一行文本是否像序列。

    剥前缀后校验长度、大小写形态、词形与氨基酸占比；三字母写法放宽比例门槛。
    单独成行的全小写英文词仍可能误判，属已知限制。
    """
    stripped = line.strip()
    if not stripped:
        return False
    # 剥掉 "序列:" / "Sequence:" 这类前缀后再判断
    body = _SEQ_PREFIX_RE.sub("", stripped)
    if not _case_is_plausible(body) or not _word_shape_is_plausible(body):
        return False
    norm, _ = normalize_candidate(body)
    if not (min_len <= len(norm) <= max_len):
        return False
    # 三字母写法已由连字符证据/覆盖率筛过，不再走比例判据。
    if _should_convert_three_letter(body):
        return True
    if len(norm) < _SEQ_LIKE_MIN_LEN:
        return False
    # 分母只数 ASCII 字母，并先剥掉全部修饰标注（末端修饰 + PEG），
    # 否则数字与修饰会拉低占比。
    body_letters = _ascii_letter_count(_sequence_body(body))
    if body_letters == 0:
        return False
    return len(norm) / body_letters >= _SEQ_LIKE_MIN_RATIO


def _whole_text_looks_like_sequence(text: str, seq_len: int) -> bool:
    """全文兜底的准入门槛：大小写形态与词形像序列，且氨基酸字母占比 ≥ 90%。

    没有门槛时整段设计理由会被拼成虚构序列并拿到奖励。
    """
    if not _case_is_plausible(text) or not _word_shape_is_plausible(text):
        return False
    body_letters = _ascii_letter_count(_sequence_body(text))
    if body_letters == 0:
        return False
    return seq_len / body_letters >= _WHOLE_TEXT_MIN_RATIO


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def extract_sequence(
    text: str,
    *,
    min_len: int = MIN_PEPTIDE_LEN,
    max_len: int = MAX_PEPTIDE_LEN,
    expect_cyclic: bool = False,
) -> ParsedSequence:
    """从模型输出中抽取序列，按模块文档中的顺序逐级回退。

    `expect_cyclic=True` 时长度下限取 4（环肽常见 5-8 残基）。
    强信号路径只要求 `min_len`；启发式路径另有 `_SEQ_LIKE_MIN_LEN=5` 的门槛。
    """
    if not text or not text.strip():
        return ParsedSequence(ok=False, reason="empty_output")

    lo = 4 if expect_cyclic else min_len
    lines = text.splitlines()

    # ---- 策略 1：### 设计序列 强标记 ----
    for i, line in enumerate(lines):
        if SEQ_OPEN in line:
            # 标记后面的第一行非空内容就是序列（模板保证了这一点）
            for follow in lines[i + 1 : i + 4]:
                if not follow.strip():
                    continue
                seq, mods = normalize_candidate(follow)
                if lo <= len(seq) <= max_len:
                    return ParsedSequence(
                        ok=True, sequence=seq, strategy="marker",
                        modifications=mods,
                        trailing_text="\n".join(lines[i + 2 :]),
                    )
                break  # 只检查紧邻的第一行非空内容，避免把"理由"当序列
            # 标记在但内容不合格，继续往下回退

    # ---- 策略 2：XML 式标签 ----
    m = _XML_TAG_RE.search(text)
    if m:
        seq, mods = normalize_candidate(m.group(1))
        if lo <= len(seq) <= max_len:
            return ParsedSequence(ok=True, sequence=seq, strategy="xml_tag",
                                  modifications=mods)

    # ---- 策略 3：Markdown 代码块 ----
    for block in _CODE_FENCE_RE.findall(text):
        seq, mods = normalize_candidate(block)
        if lo <= len(seq) <= max_len:
            # 代码块里可能既有序列又有别的行，再逐行确认一次
            for line in block.splitlines():
                line_seq, line_mods = normalize_candidate(line)
                if lo <= len(line_seq) <= max_len and _looks_like_sequence(
                    line, min_len=lo, max_len=max_len
                ):
                    return ParsedSequence(
                        ok=True, sequence=line_seq, strategy="code_fence",
                        modifications=line_mods,
                    )
            return ParsedSequence(ok=True, sequence=seq, strategy="code_fence",
                                  modifications=mods)

    # ---- 策略 4：逐行扫描找"最像序列"的那一行 ----
    candidates: list[tuple[int, str, list[str]]] = []
    for line in lines:
        if not _looks_like_sequence(line, min_len=lo, max_len=max_len):
            continue
        seq, mods = normalize_candidate(line)
        if lo <= len(seq) <= max_len:
            # 打分：含模糊残基（X/B/Z）越少越好，因为那些通常是噪声
            ambiguity = sum(1 for c in seq if c not in STANDARD_AA)
            score = len(seq) - ambiguity * 3
            candidates.append((score, seq, mods))
    if candidates:
        candidates.sort(key=lambda t: -t[0])
        _, seq, mods = candidates[0]
        return ParsedSequence(ok=True, sequence=seq, strategy="line_scan",
                              modifications=mods)

    # ---- 策略 5：全文兜底 ----
    # 准入判据见 _whole_text_looks_like_sequence：没有它的话，
    # 一段英文设计理由会被整段捡走、拼成一条虚构的"序列"。
    seq, mods = normalize_candidate(text)
    if lo <= len(seq) <= max_len and _whole_text_looks_like_sequence(text, len(seq)):
        return ParsedSequence(ok=True, sequence=seq, strategy="whole_text",
                              modifications=mods)

    # ---- 全部失败：给出可定位的原因 ----
    reason = "too_long" if len(seq) > max_len else (
        "too_short" if 0 < len(seq) < lo else "no_sequence_found"
    )
    return ParsedSequence(ok=False, reason=reason, strategy="failed")


def extract_reasoning(text: str) -> str:
    """抽取"设计理由"文本，用于人工审阅；不参与奖励计算。"""
    from pepdesign.prompts.templates import SEQ_CLOSE

    idx = text.find(SEQ_CLOSE)
    if idx < 0:
        return ""
    rest = text[idx + len(SEQ_CLOSE) :]
    end = rest.find("### 结束")
    return (rest[:end] if end >= 0 else rest).strip()


def parse_stats(results: list[ParsedSequence]) -> dict[str, float | int]:
    """汇总一批解析结果的统计量，供训练日志使用。"""
    n = len(results)
    if n == 0:
        return {}
    strategy_dist: dict[str, int] = {}
    reason_dist: dict[str, int] = {}
    for r in results:
        strategy_dist[r.strategy] = strategy_dist.get(r.strategy, 0) + 1
        if not r.ok:
            reason_dist[r.reason] = reason_dist.get(r.reason, 0) + 1
    ok = sum(1 for r in results if r.ok)
    lengths = [len(r.sequence) for r in results if r.ok]
    return {
        "total": n,
        "ok": ok,
        "fail_rate": round(1 - ok / n, 4),
        "mean_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "strategy_distribution": strategy_dist,
        "fail_reasons": reason_dist,
    }
