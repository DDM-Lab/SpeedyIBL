from speedyibl import Agent 
import random
import tqdm
import pprint
import numpy as np
import time 

# DEFAULT_PARTICIPANTS = 1000
DEFAULT_PARTICIPANTS = 1000
DEFAULT_NOISE = 0.25
DEFAULT_TEMPERATURE = 1.0
DEFAULT_DECAY = 0.5
DEFAULT_MISMATCH_PENALTY = 2.5
DEFAULT_OUTPUT_FILE = "insider-speedyIBL-data.csv"

TARGETS = [ [ { "payment": 2, "penalty":  -1, "monitored_probability": 0.22 },
              { "payment": 8, "penalty":  -5, "monitored_probability": 0.51 },
              { "payment": 9, "penalty":  -9, "monitored_probability": 0.42 },
              { "payment": 9, "penalty": -10, "monitored_probability": 0.40 },
              { "payment": 2, "penalty":  -6, "monitored_probability": 0.08 },
              { "payment": 5, "penalty":  -5, "monitored_probability": 0.36 } ],
            [ { "payment": 5, "penalty":  -3, "monitored_probability": 0.41 },
              { "payment": 8, "penalty":  -5, "monitored_probability": 0.48 },
              { "payment": 7, "penalty":  -6, "monitored_probability": 0.41 },
              { "payment": 8, "penalty":  -9, "monitored_probability": 0.37 },
              { "payment": 5, "penalty":  -7, "monitored_probability": 0.27 },
              { "payment": 2, "penalty":  -4, "monitored_probability": 0.05 } ],
            [ { "payment": 3, "penalty":  -3, "monitored_probability": 0.30 },
              { "payment": 9, "penalty":  -4, "monitored_probability": 0.60 },
              { "payment": 6, "penalty":  -6, "monitored_probability": 0.40 },
              { "payment": 5, "penalty":  -8, "monitored_probability": 0.29 },
              { "payment": 3, "penalty":  -6, "monitored_probability": 0.20 },
              { "payment": 2, "penalty":  -2, "monitored_probability": 0.20 } ],
            [ { "payment": 4, "penalty":  -3, "monitored_probability": 0.37 },
              { "payment": 6, "penalty":  -3, "monitored_probability": 0.51 },
              { "payment": 7, "penalty":  -7, "monitored_probability": 0.40 },
              { "payment": 5, "penalty": -10, "monitored_probability": 0.24 },
              { "payment": 5, "penalty":  -9, "monitored_probability": 0.26 },
              { "payment": 3, "penalty":  -4, "monitored_probability": 0.23 } ] ]

COVERAGE = [ [ { 2, 6 }, { 2, 4 }, { 2, 5 }, { 2, 4 }, { 1, 3 },
               { 2, 4 }, { 1, 3 }, { 1, 3 }, { 2, 4 }, { 2, 6 },
               { 2, 6 }, { 2, 4 }, { 1, 3 }, { 2, 4 }, { 2, 4 },
               { 1, 3 }, { 3, 6 }, { 2, 4 }, { 2, 4 }, { 3, 6 },
               { 1, 3 }, { 2, 4 }, { 3, 6 }, { 2, 4 }, { 1, 3 } ],
             [ { 2, 5 }, { 1, 3 }, { 1, 3 }, { 3, 6 }, { 1, 3 },
               { 2, 4 }, { 1, 3 }, { 2, 4 }, { 1, 3 }, { 1, 4 },
               { 1, 3 }, { 1, 3 }, { 2, 5 }, { 1, 3 }, { 1, 3 },
               { 1, 3 }, { 2, 5 }, { 2, 4 }, { 2, 4 }, { 1, 3 },
               { 1, 3 }, { 2, 4 }, { 2, 4 }, { 3, 6 }, { 2, 5 } ],
             [ { 2, 5 }, { 3, 6 }, { 2, 4 }, { 2, 5 }, { 2, 5 },
               { 2, 6 }, { 2, 6 }, { 1, 3 }, { 2, 4 }, { 1, 3 },
               { 2, 4 }, { 1, 3 }, { 1, 3 }, { 2, 6 }, { 2, 5 },
               { 1, 3 }, { 2, 4 }, { 1, 3 }, { 2, 4 }, { 2, 5 },
               { 2, 4 }, { 2, 4 }, { 2, 6 }, { 1, 3 }, { 2, 4 } ],
             [ { 2, 5 }, { 1, 4 }, { 3, 6 }, { 2, 6 }, { 1, 3 },
               { 1, 4 }, { 1, 3 }, { 2, 5 }, { 2, 6 }, { 1, 3 },
               { 1, 3 }, { 3, 6 }, { 2, 4 }, { 1, 4 }, { 1, 4 },
               { 1, 3 }, { 1, 3 }, { 1, 4 }, { 1, 3 }, { 2, 5 },
               { 3, 6 }, { 1, 3 }, { 1, 3 }, { 3, 6 }, { 1, 4 } ] ]

TRAINING_COVERAGE = [ { 2, 5 }, { 2, 4 }, { 1 , 3 }, { 1, 3 }, { 1, 3 } ]

SIGNALS = [ [ { 3, 4 }, { 3, 6 }, { 3, 6 }, { 3, 5, 6 }, { 2, 6 },
              { 3, 6 }, { 2, 4}, { 2, 6 }, { 3, 6 }, { 1, 3, 4 },
              { 3, 4 }, { 1, 3 }, { 4, 6 }, { 5}, { 3, 6 },
              { 2, 4 }, { 5 }, { 3 }, { 6 }, { 2, 4 },
              { 2, 4 }, set(), {2, 4, 5 }, { 3 }, { 5, 6 } ],
            [ { 3, 4 }, { 2, 4 }, { 2, 4, 5 }, { 4, 5 }, { 4, 5 },
              { 1, 3, 6 }, { 2 }, { 3 }, { 5 }, set(),
              { 2, 5 }, { 2, 5 }, {3, 4 }, { 2, 5 }, { 2, 4, 5 },
              { 4, 5 }, { 3, 4 }, { 3, 5, 6 }, { 1, 5}, { 2, 5 },
              { 2 }, { 1, 5 }, { 1, 3, 5 }, { 4 }, { 1, 3, 4, 6 } ],
            [ { 1, 3, 6 }, { 2, 4 }, set(), { 1, 3, 4 }, { 3 },
              { 1, 4, 5 }, { 5 }, { 2, 4}, { 1, 3, 5 }, set(),
              { 1, 3, 5 }, { 2 }, { 2, 4, 5 }, { 5 }, { 3, 4 },
              { 2, 4, 5, 6 }, { 1, 3, 5 }, { 2, 4, 6 }, { 1, 3 }, { 1, 4 },
              { 5 }, {3 }, set(), { 2, 5, 6 }, { 1, 3, 5, 6 } ],
            [ { 6 }, { 3 }, { 2, 4 }, { 4, 5}, { 6 },
              { 3, 5 }, { 4 }, { 3, 4, 6 }, { 1, 3, 4, 5 }, { 2, 4, 6 },
              {4, 5 }, { 2, 5 }, { 1, 5, 6 }, { 2, 3, 6 }, { 2, 3 },
              { 5 }, { 2, 4, 5, 6 }, { 2, 3, 5, 6 }, { 2, 4, 5 }, { 1, 3, 4, 6 },
              { 2, 4, 5 }, { 4, 5 }, { 4 }, { 4, 5 }, { 3, 5, 6 } ] ]

TRAINING_SIGNALS = [ { 3, 4 }, {1, 3, 6 }, { 5 }, { 2, 5 }, {2, 4, 5} ]

for clist, slist in zip(COVERAGE, SIGNALS):
    for c, s in zip(clist, slist):
        s.update(c)

TARGET_COUNT = len(TARGETS[0])
BLOCKS = len(TARGETS)
TRIALS = len(COVERAGE[0])

def reset_agent(a,
                noise=DEFAULT_NOISE,
                temperature=DEFAULT_TEMPERATURE,
                decay=DEFAULT_DECAY,
                mismatch_penalty=DEFAULT_MISMATCH_PENALTY):
    a.reset()
    a.noise = noise
    a.temperature = temperature
    a.decay = decay
    a.mismatchPenalty = mismatch_penalty

def run(output_file=DEFAULT_OUTPUT_FILE, participants=DEFAULT_PARTICIPANTS):
    selection_agent = Agent(default_utility=None)
    attack_agent = Agent(default_utility=None)
    attacks = [0] * BLOCKS * TRIALS
    with open(output_file, "w") as f:
        print("Subject,Block,Trial,Running_Trial,Selected,Warning,Covered,Action,Outcome,Cum_Outcome,time", file=f)
        for p in tqdm.tqdm(range(participants)):
            total = 0
            total_time = 0
            reset_agent(selection_agent)
            selection_agent.similarity([0,1], lambda x, y: 1 - abs(x - y) / 10)
            selection_agent.similarity([2], lambda x, y: 1 - abs(x -y))
            # selection_agent.similarity(None, lambda x, y: 1)
            attack_agent.reset()
            dup = random.randrange(5)
            for i in range(5):
                n = random.randrange(TARGET_COUNT)
                # x = TARGETS[1][n]
                x = TARGETS[1][n]
                covered = n + 1 in TRAINING_COVERAGE[i]
                selection_agent.prepopulate(((x["payment"],
                                                x["penalty"],
                                                x["monitored_probability"]), i+1),
                                                x["penalty" if covered else "payment"])
                attack_agent.prepopulate((True, n + 1 in TRAINING_SIGNALS[i]),x["penalty" if covered else "payment"])
                if i == dup:
                    # x = TARGETS[1][5]
                    selection_agent.prepopulate(((x["payment"],
                                                x["penalty"],
                                                x["monitored_probability"]), 6),
                                                x["penalty" if covered else "payment"])
            attack_agent.prepopulate((False,False),0)
            attack_agent.prepopulate((False,True),0)
            attack_agent.prepopulate((True,False),10)
            attack_agent.prepopulate((False,True),5)
            # selection_agent.instances()
            
            for b in range(BLOCKS):
                sds = [ ((x["payment"],
                                        x["penalty"],
                                        x["monitored_probability"]), i+1)
                        for x, i in zip(TARGETS[b], range(TARGET_COUNT)) ]

                for t in range(TRIALS):
                    start_time = time.time()
                    selected = selection_agent.choose(sds)[1]
                    warned = selected in SIGNALS[b][t]
                    pmnt = TARGETS[b][selected - 1]["payment"]
                    attack = attack_agent.choose([(True, warned),
                                                 (False, warned)])[0]
                    covered = selected in COVERAGE[b][t]
                    if not attack:
                        payoff = 0
                    else:
                        payoff = TARGETS[b][selected - 1]["penalty" if covered else "payment"]
                        attacks[b * 25 + t] += 1
                    total += payoff
                    attack_agent.respond(payoff)
                    selection_agent.respond(payoff)

                    total_time += time.time() - start_time
                    print(f"{p+1},{b+1},{t+1},{b*25+t+1},{selected},{int(warned)},{int(covered)},{int(attack)},{payoff},{total},{total_time}", file=f)

def main():
    run()


if __name__ == "__main__":
    main()
