"""提示词层：指令模板与模型输出解析。

模板规定输出格式，解析器按同一约定读取；两者是同一份契约，放在一起修改和测试，
避免只改一侧而使另一侧失效。
"""

from pepdesign.prompts import parser, templates

__all__ = ["parser", "templates"]
