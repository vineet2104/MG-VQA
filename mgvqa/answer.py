"""Answer submission only. Correctness belongs in verification.py, not tools."""


def answer_question(response):
    return {key: response.get(key) for key in ('answer', 'point', 'ratio')}
