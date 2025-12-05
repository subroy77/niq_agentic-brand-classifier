from agent.pipeline_agentic import classify_single_query


def test_classify_single_query_shape():
    # This is a very light shape test. It does not assert specific labels.
    query = "Compare Samsung and LG smart TVs for HDR performance"
    result = classify_single_query(query)
    d = result.to_dict()

    assert "brand" in d
    assert "category" in d
    assert "sub_category" in d
    assert isinstance(d["category_confidence"], float)
    assert isinstance(d["sub_category_confidence"], float)
