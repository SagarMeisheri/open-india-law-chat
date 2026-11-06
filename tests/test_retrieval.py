from agents.legal_agent import retrieval as r


def test_state_names_map_to_stored_slugs():
  vals, rejected = r.normalize_jurisdictions(
      ['Madhya Pradesh', 'Jammu and Kashmir', 'Dadra and Nagar Haveli',
       'Andaman and Nicobar Islands', 'central', 'madhya-pradesh', 'Atlantis'])
  assert vals == ['madhya-pradesh', 'jammu-kashmir', 'dadra-nagar-haveli',
                  'andaman-nicobar', 'central']
  assert rejected == ['Atlantis']


def test_every_state_slug_is_a_value_seen_in_the_real_data():
  seen = {  # from the probe's facet over acts_india
      'andaman-nicobar', 'andhra-pradesh', 'arunachal-pradesh', 'assam', 'bihar',
      'chandigarh', 'chhattisgarh', 'dadra-nagar-haveli', 'delhi', 'goa',
      'gujarat', 'haryana', 'himachal-pradesh', 'jammu-kashmir', 'jharkhand',
      'karnataka', 'kerala', 'ladakh', 'lakshadweep', 'madhya-pradesh',
      'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha',
      'puducherry', 'punjab', 'rajasthan', 'sikkim', 'tamil-nadu', 'telangana',
      'tripura', 'uttar-pradesh', 'uttarakhand', 'west-bengal'}
  assert set(r.STATE_SLUGS.values()) == seen


def test_in_force_requires_act_and_section_in_force():
  f = r.build_filter({'status': 'in_force'})
  keys = {c.key: c.match.value for c in f.must}
  assert keys == {'act_status': 'in_force', 'section_status': 'in_force'}


def test_repealed_matches_act_or_section():
  f = r.build_filter({'status': 'repealed'})
  inner = f.must[0].should
  assert {c.key for c in inner} == {'act_status', 'section_status'}


def test_no_filters_means_no_filter():
  assert r.build_filter({}) is None
