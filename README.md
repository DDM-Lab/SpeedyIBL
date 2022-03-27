# A customized version of official SpeedyIBL

> Used for CybORG defender modeling

- Different `instance_history` organization:
    - It would be helpful to support various instance_history organization 
    - It would be more user friendly to use more self-explainable variable names rather than `a`, `s`, and `tmp`
- Removed similarity calculation for exact-matching
- Different `equal_delay_feedback` function:
    - Specify `new_reward` and a list of instances to update (`episode_history`)
    - It would be helpful if SpeedyIBL can provide options for equal credit assignment window size and keep track of the instances to be updated