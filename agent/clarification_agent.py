import dspy


class ClarificationSignature(dspy.Signature):
    """
    When classification is uncertain or inconsistent, this agent
    generates a short clarifying question to ask the user.
    """
    query = dspy.InputField()
    issue = dspy.InputField(desc="Why the system is uncertain (free text)")
    question = dspy.OutputField(
        desc="One short clarifying question to ask the user."
    )


clarification_module = dspy.Predict(ClarificationSignature)
