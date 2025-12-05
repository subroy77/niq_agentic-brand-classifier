import dspy


class ValidationSignature(dspy.Signature):
    """
    Validation agent that checks if (brand, category, sub_category)
    are consistent with the original query.
    """
    query = dspy.InputField()
    brand = dspy.InputField()
    category = dspy.InputField()
    sub_category = dspy.InputField()

    is_consistent = dspy.OutputField(
        desc='Return "yes" if labels are consistent, "no" otherwise.'
    )
    reason = dspy.OutputField(
        desc="Short explanation if labels appear inconsistent or low-quality."
    )


validator_module = dspy.Predict(ValidationSignature)
