"""OpenCompass's option extractor, copied verbatim so the quick utility check
scores PIQA exactly as OpenCompass's default piqa_gen config does.

Source: opencompass 0.5.4, opencompass/utils/text_postprocessors.py
(Apache License 2.0, https://github.com/open-compass/opencompass).

The only edit: its f-strings are written as raw rf-strings, which gives
identical patterns without Python's invalid-escape warnings. Otherwise
unchanged, including the Chinese patterns and the missing comma after the
'故选' pattern. English replies fall through to the 'answer is' / 'ANSWER:' /
bare-letter patterns near the end of the list.
"""
import re


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        rf'答案是?\s*([{options}])',
        rf'答案是?\s*：\s*([{options}])',
        rf'答案是?\s*:\s*([{options}])',
        rf'答案选项应?该?是\s*([{options}])',
        rf'答案选项应?该?为\s*([{options}])',
        rf'答案应该?是\s*([{options}])',
        rf'答案应该?选\s*([{options}])',
        rf'答案选项为?\s*：\s*([{options}])',
        rf'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'答案选项是?\s*:\s*([{options}])',
        rf'答案为\s*([{options}])',
        rf'答案选\s*([{options}])',
        rf'选择?\s*([{options}])',
        rf'故选?\s*([{options}])'
        rf'只有选?项?\s?([{options}])\s?是?对',
        rf'只有选?项?\s?([{options}])\s?是?错',
        rf'只有选?项?\s?([{options}])\s?不?正确',
        rf'只有选?项?\s?([{options}])\s?错误',
        rf'说法不?对选?项?的?是\s?([{options}])',
        rf'说法不?正确选?项?的?是\s?([{options}])',
        rf'说法错误选?项?的?是\s?([{options}])',
        rf'([{options}])\s?是正确的',
        rf'([{options}])\s?是正确答案',
        rf'选项\s?([{options}])\s?正确',
        rf'所以答\s?([{options}])',
        rf'所以\s?([{options}][.。$]?$)',
        rf'所有\s?([{options}][.。$]?$)',
        rf'[\s，：:,]([{options}])[。，,\.]?$',
        rf'[\s，,：:][故即]([{options}])[。\.]?$',
        rf'[\s，,：:]因此([{options}])[。\.]?$',
        rf'[是为。]\s?([{options}])[。\.]?$',
        rf'因此\s?([{options}])[。\.]?$',
        rf'显然\s?([{options}])[。\.]?$',
        rf'答案是\s?(\S+)(?:。|$)',
        rf'答案应该是\s?(\S+)(?:。|$)',
        rf'答案为\s?(\S+)(?:。|$)',
        rf'(?i)ANSWER\s*:\s*([{options}])',
        rf'[Tt]he answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        rf'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        rf'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        rf'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        rf'^选项\s?([{options}])',
        rf'^([{options}])\s?选?项',
        rf'(\s|^)[{options}][\s。，,：:\.$]',
        rf'1.\s?(.*?)$',
        rf'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        rf'([{options}]):',
        rf'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''
