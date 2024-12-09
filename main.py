import numpy as np
import pm4py
import pandas as pd
from jinja2 import Template
from skeleton import skeleton
import pickle

def add_row_position(group):
    group['Row_Position'] = range(len(group))
    return group

def extract_timestamp_features(group):
        timestamp_col = 'timestamps'
        group = group.sort_values(timestamp_col, ascending=True)
        # end_date = group[timestamp_col].iloc[-1]
        start_date = group[timestamp_col].iloc[0]

        timesincelastevent = group[timestamp_col].diff()
        timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
        group["timesincelastevent"] = timesincelastevent.apply(
            lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

        elapsed = group[timestamp_col] - start_date
        elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
        group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds
        return group


def history_conversion(csv_log):
    group_csv = csv_log.groupby('stay_id', sort=False)#.agg({'activity': lambda x: list(x)})
    list_label = []
    list_seq = []
    list_len_prefix = []
    for group_name, group_data in group_csv:
        event_dict_hist = {}
        event_text = ''
        len_prefix = 1
        list_subtrace = []
        check = False
        group_data['Next_Value'] = group_data.groupby('stay_id', sort=False)['activity'].shift(-1)
        group_data.loc[group_data.index[-1], 'Next_Value'] = 'Discharge from the ED'

        group_data.to_csv('group_data.csv', index=False)
        group_data = group_data.groupby('stay_id').apply(add_row_position).reset_index(drop=True)


        for index, row in group_data.iterrows():
            for v in skeleton['feature']:
                value = row[v]
                if isinstance(value, str):
                    event_dict_hist[v] = value
                else:
                    event_dict_hist[v] = value

            event_template = Template(skeleton[row['activity']])

            if row['activity'] != 'Medicine reconciliation' and row['activity'] != 'Medicine dispensations' and row['activity'] != 'Discharge from the ED':
                list_label.append(group_data['disposition'].tail(1).values[0])
                event_text = event_text + event_template.render(event_dict_hist) + '. '
                list_seq.append(event_text)
                list_len_prefix.append(len_prefix)
                len_prefix = len_prefix + 1
            elif row['activity'] == 'Medicine dispensations' and check==False:
                last_b_index = row['Row_Position']
                while last_b_index < len(group_data) and group_data.loc[last_b_index, 'activity'] == 'Medicine dispensations':
                    list_subtrace.append(group_data.loc[last_b_index, 'name'])
                    last_b_index += 1
                my_string = ", ".join(str(element) for element in list_subtrace)
                event_text = event_text + event_template.render(event_dict_hist) + ' ' + my_string + '. '
                list_label.append(group_data['disposition'].tail(1).values[0])
                list_subtrace = []
                check = True
                list_seq.append(event_text)
                list_len_prefix.append(len_prefix)
                len_prefix = len_prefix + 1
            elif row['activity'] == 'Medicine reconciliation' and check==False:
                last_b_index = row['Row_Position']
                while last_b_index  < len(group_data) and group_data.loc[last_b_index, 'activity'] == 'Medicine reconciliation':
                    list_subtrace.append(group_data.loc[last_b_index, 'name_etc'])
                    last_b_index += 1
                list_label.append(group_data['disposition'].tail(1).values[0])
                my_string = ", ".join(str(element) for element in list_subtrace)
                event_text = event_text + event_template.render(event_dict_hist) + ' ' + my_string + '. '
                list_subtrace = []
                check = True
                list_seq.append(event_text)
                list_len_prefix.append(len_prefix)
                len_prefix = len_prefix + 1
            elif row['activity'] == 'Discharge from the ED' and check==False:
                last_b_index = row['Row_Position']
                while last_b_index < len(group_data) and group_data.loc[last_b_index, 'activity'] == 'Discharge from the ED':
                    list_subtrace.append(group_data.loc[last_b_index, 'icd_title'])
                    last_b_index += 1
                list_label.append(group_data['disposition'].tail(1).values[0])
                my_string = ", ".join(str(element) for element in list_subtrace)
                event_text = event_text + event_template.render(event_dict_hist) + ' ' + my_string + '.'
                list_subtrace = []
                check = True
                list_seq.append(event_text)
                list_len_prefix.append(len_prefix)
                len_prefix = len_prefix + 1

            if row['activity'] != row['Next_Value']:
                check = False
            else:
                check = True
    return list_seq, list_label

if __name__ == "__main__":
    csv_log = pd.read_csv('mimicel.csv')

    csv_log['timestamps'] = pd.to_datetime(csv_log['timestamps'])

    csv_log = csv_log.groupby('stay_id', group_keys=False).apply(extract_timestamp_features)
    csv_log = csv_log.reset_index(drop=True)
    csv_log['timesincecasestart'] = (csv_log['timesincecasestart'])  # .round(3)
    csv_log['timesincecasestart'] = csv_log['timesincecasestart'].astype(int)
    csv_log['name_etc'] = csv_log['name'] + ' that is ' + csv_log['etcdescription']
    csv_log = csv_log.fillna('N/A')

    grouped = csv_log.groupby("stay_id")
    start_timestamps = grouped["timestamps"].min().reset_index()
    start_timestamps = start_timestamps.sort_values("timestamps", ascending=True, kind="mergesort")
    train_ids = list(start_timestamps["stay_id"])[:int(0.66 * len(start_timestamps))]
    train = csv_log[csv_log["stay_id"].isin(train_ids)].sort_values("timestamps", ascending=True,
                                                                              kind='mergesort')
    test = csv_log[~csv_log["stay_id"].isin(train_ids)].sort_values("timestamps", ascending=True,
                                                                              kind='mergesort')

    list_seq_train, list_label_train = history_conversion(train)
    list_seq_test, list_label_test = history_conversion(test)

    with open('mimic_train.pkl', 'wb') as f:
        pickle.dump(list_seq_train, f)

    with open('mimic_test.pkl', 'wb') as f:
        pickle.dump(list_seq_test, f)

    with open('mimic_label_train.pkl', 'wb') as f:
        pickle.dump(list_label_train, f)

    with open('mimic_label_test.pkl', 'wb') as f:
        pickle.dump(list_label_test, f)


