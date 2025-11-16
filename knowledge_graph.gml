graph [
  directed 1
  node [
    id 0
    label "P001"
    type "Problem"
  ]
  node [
    id 1
    label "I003"
    type "Intervention"
  ]
  node [
    id 2
    label "T01"
    type "Tag"
  ]
  edge [
    source 0
    target 1
    relation "addresses"
    weight 0.8
    evidence_id "E001"
  ]
  edge [
    source 0
    target 2
    relation "related_to"
    weight 1.0
    evidence_id "E002"
  ]
]
