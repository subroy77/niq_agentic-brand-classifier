import dspy


class SubCategorySignature(dspy.Signature):
    """
    Classify a user query into a more granular sub-category under a
    given high-level category.

    No fixed taxonomy is used here. The model should invent short,
    business-meaningful sub-category labels. If unsure, it must use "Other".
    """
    query = dspy.InputField(desc="The original user query or prompt")
    category = dspy.InputField(desc="The high-level category already inferred")
    sub_category = dspy.OutputField(
        desc="Short, human-readable sub-category label (string)"
    )
    confidence = dspy.OutputField(
        desc="Model's confidence between 0 and 1"
    )


sub_category_module = dspy.Predict(SubCategorySignature)
