# There is not a vehicle nearby and therefore cannot include dx and dy values in the observations
MISSING_NEARBY_AGENT_VALUE = -1
LSTM_PADDING_VALUE = -1

DEFAULT_SCENARIO = "appershofen"  # scenario to use for the default value
DEFAULT_CLUSTER = "Aggressive"
DEFAULT_DECELERATION_VALUE = 0

# These agents are out of the road or have wrong velocity begin '0a37851d-eb39-4409-a1ad-c1e6ec313f91' has wrong later_acc
REMOVED_AGENTS = ['29c74d22-9aa7-442d-b3ca-8a710ef26185', '88849c8f-5765-4898-8833-88589b72b0bd',
                  'c6025d47-2d30-419b-8b18-48ec83a3619c', '0a37851d-eb39-4409-a1ad-c1e6ec313f91','0754a583-8ba1-432f-8272-d6a1b911e689','61c25a4c-e3c6-4dee-83a4-fbd80376ce52',
                  '598a6e25-bc9e-4b0f-bdb1-692f73d3a191','648dd669-20d1-4512-ab1c-bfb4b6b2bc71','bec34161-bddd-4778-9bab-cd4de2b7b8d0',
                  '329b7035-9e9b-4fbf-9cae-4ae562bdd8de','99f93eda-2660-4547-a768-4cdf1cf6913e']
