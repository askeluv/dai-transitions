import pandas as pd
from functools import reduce

ORIGINAL_STATES = ['born', 'unsafe', 'safe again', 'wiped', 'bitten', 'shut']
STATES = ['safe', 'unsafe', 'wiped', 'bitten', 'shut']
ABSORBING_STATES = ['wiped', 'bitten', 'shut']
LAST_TIMESTAMP = 1549495867 # last timestamp observed (i.e. end of the observation period)

def sorted_states_with_timestamps(row):
    "Return a sorted list of states w/ timestamps, given a row from the original dataset."
    d = row.to_dict()
    states = [s for s in d.items() if pd.notnull(s[1])]
    return sorted(states, key=lambda x: x[1])

def simplify_state(s):
    "Collapse 'born' and 'safe again' states into one state: 'safe'."
    if s in ('born', 'safe again'):
        return 'safe'
    else:
        return s

def seconds_spent_in_each_state(seq):
    "Calculate seconds spent in each state for a given sequence."
    result = []
    for (i, (s, ts)) in enumerate(seq):
        if i + 1 == len(seq):
            time_spent = LAST_TIMESTAMP - ts
        else:
            next_ts = seq[i + 1][1]
            time_spent = next_ts - ts
        s = simplify_state(s)
        result.append((s, ts, time_spent))
    return result

def seconds_spent_with_end_state(seq):
    "Calculate seconds spent in each state for a given sequence."
    result = []
    end_state = '<end>'
    for (i, (s, ts)) in enumerate(seq):
        s = simplify_state(s)
        if i + 1 == len(seq):
            if s in ABSORBING_STATES:
                return result
            time_spent = LAST_TIMESTAMP - ts
            next_state = end_state
        else:
            next_ts = seq[i + 1][1]
            next_state = simplify_state(seq[i + 1][0])
            time_spent = next_ts - ts
        result.append((s, ts, time_spent, next_state))
    return result

def dataframe_to_sequences(df):
    "Convert the original dataframe into a list of state sequences."
    sorted_sequences_with_timestamps = list(df[ORIGINAL_STATES].apply(sorted_states_with_timestamps, axis=1))
    return list(map(seconds_spent_in_each_state, sorted_sequences_with_timestamps))

def dataframe_to_sequences_with_end_state(df):
    "Convert the original dataframe into a list of state sequences."
    sorted_sequences_with_timestamps = list(df[ORIGINAL_STATES].apply(sorted_states_with_timestamps, axis=1))
    return list(map(seconds_spent_with_end_state, sorted_sequences_with_timestamps))

def transitions(seq):
    "Create a list of state sequences, where the next state is added to each state."
    result = []
    for (i, (s, ts, seconds)) in enumerate(seq):
        if i + 1 == len(seq):
            return result
        else:
            next_state = seq[i + 1][0]
            result.append((s, next_state, 1))
    return result

def total_seconds_per_state(sequences):
    "Calculates the total seconds spent per state."
    flat_sequences = reduce(lambda x, y: x + y, sequences)
    return pd.DataFrame(flat_sequences).groupby(0).sum()[2]

def transition_counts(sequences):
    transition_pairs = list(map(transitions, sequences))
    flat_transition_pairs = reduce(lambda x, y: x+y, transition_pairs)
    for s in STATES:
        flat_transition_pairs.append((s, s, 0))
    pairs_df = pd.DataFrame(flat_transition_pairs).groupby([0, 1]).sum().reset_index()
    counts = pairs_df.pivot_table(index=0, columns=1)[2]
    counts.index.name = None
    counts.columns.name = None
    return counts

def transition_probabilities(sequences, seconds_per_time_unit=1):
    "Calculates a matrix Q_ij containing probability of transitioning from state i to j."
    counts = transition_counts(sequences)
    seconds = total_seconds_per_state(sequences)
    matrix = counts.copy()
    for row in STATES:
        matrix.loc[row] = counts.loc[row] / (seconds[row] / seconds_per_time_unit)
        row_sum = matrix.loc[row].sum()
        matrix[row][row] = 1 - row_sum # p(i | i) = 1 - sum[p(i | j, i!=j)]
    matrix.fillna(0, inplace=True)
    return matrix.loc[STATES][STATES]

def transition_rates(sequences, seconds_per_time_unit=1):
    "Calculates a matrix Λ_ij containing transition rates from state i to j."
    counts = transition_counts(sequences)
    seconds = total_seconds_per_state(sequences)
    matrix = counts.copy()
    for row in STATES:
        matrix.loc[row] = counts.loc[row] / (seconds[row] / seconds_per_time_unit)
        row_sum = matrix.loc[row].sum()
        if row_sum > 0:
            matrix[row][row] = -row_sum # λ_ij = -λ_i
    matrix.fillna(0, inplace=True)
    return matrix.loc[STATES][STATES]

def sequence_distribution(sequences):
    "Probability distribution over observed sequences."
    return pd.Series([' -> '.join([s[0] for s in sequence]) for sequence in sequences]).value_counts(normalize=1)

def time_spent_before_state_change_distribution(sequences_with_next_state):
    "Probability distribution over time spent in state i before changing to state j."
    swns = reduce(lambda x,y : x + y, sequences_with_next_state)
    swns_df = pd.DataFrame(swns, columns=['from_state', 'ts', 'seconds_spent', 'to_state'])
    piv = swns_df.pivot_table(index='from_state', columns='to_state', values='seconds_spent', aggfunc=pd.np.sum)
    return piv / piv.sum().sum()
