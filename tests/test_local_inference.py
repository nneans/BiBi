from bibi_reviews.local_inference import choose_states, make_version


def test_abstain_and_preserve_multiple_reasons():
    candidates=[{'topic_id':'bones','aspect':'quality','sentiment':'negative','score':.85},
                {'topic_id':'taste','aspect':'taste','sentiment':'positive','score':.90},
                {'topic_id':'delivery','aspect':'delivery','sentiment':'negative','score':.4}]
    out=choose_states(candidates,.7)
    assert {x['topic_id'] for x in out}=={'bones','taste'}
    assert choose_states(candidates,.99)==[]


def test_deferred_wins_and_version_covers_catalogue_settings():
    rows=[{'topic_id':'taste','sentiment':'positive','score':.81},
          {'topic_id':'taste','sentiment':'deferred','score':.91}]
    assert choose_states(rows,.7)[0]['sentiment']=='deferred'
    assert make_version({'threshold':.7},[{'x':1}])!=make_version({'threshold':.8},[{'x':1}])
    assert make_version({'threshold':.7},[{'x':1}])!=make_version({'threshold':.7},[{'x':2}])


def test_roberta_with_single_segment_vocab_drops_pair_segment_ids():
    from bibi_reviews.local_inference import supported_model_inputs
    tokens={'input_ids':[2,3], 'attention_mask':[1,1], 'token_type_ids':[0,1]}
    assert 'token_type_ids' not in supported_model_inputs(tokens,1)
    assert supported_model_inputs(tokens,2)['token_type_ids']==[0,1]
