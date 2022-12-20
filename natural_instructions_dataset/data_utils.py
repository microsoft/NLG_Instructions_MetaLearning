"""
Partition natural instructions expansion dataset into train, test and dev data.

Directories train, test and dev will be created inside output_dirpath.
"""
import argparse
import json
import os
import random
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='Split NI dataset')
    parser.add_argument('--input_dirpath', type=str,
                        default='/mnt/Repos/natural-instructions-expansion/tasks/',
                        help='Path to directory with NI tasks')
    parser.add_argument('--output_dirpath', type=str,
                        default='/mnt/Data/natural_instructions_v2.5/',
                        help='Path to directory where results will be stored.')
    parser.add_argument('--limit_tasks', type=int, default=-1,
                        help='Limit the number of tasks processed. To include all tasks, '
                              'use negative number.')
    args = parser.parse_args()
    return args


def create_dirs(output_dirpath):
    """Returns partition directory names. Ensures directories exist."""
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)
    output_dirpath = os.path.join(output_dirpath, 'tasks') 
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)

    output_dirpath_train = os.path.join(output_dirpath, "train")
    output_dirpath_dev = os.path.join(output_dirpath, "dev")
    output_dirpath_test = os.path.join(output_dirpath, "test")

    if not os.path.exists(output_dirpath_train):
        os.mkdir(output_dirpath_train)
    if not os.path.exists(output_dirpath_dev):
        os.mkdir(output_dirpath_dev)
    if not os.path.exists(output_dirpath_test):
        os.mkdir(output_dirpath_test)

    return output_dirpath_train, output_dirpath_test, output_dirpath_dev


def parse_task(task_list_dict_readme, split_ratio, task_meta_data_dict, task_name, source_data):
    """Splits @source_data into partitions according to split_ratio."""
    train_data = {}
    dev_data = {}
    test_data = {}
    # Instructions have the following components: ['Categories', 'Definition',
    # 'Positive Examples', 'Negative Examples', 'Instances']
    meta_data_keys = ["Categories", "Domains", "Input_language", "Output_language", "Instruction_language"]
    meta_data_dict = {}
    
    for k in meta_data_keys:
        if k in source_data.keys():
            meta_data_dict[k] = source_data[k]
        else:
            meta_data_dict[k] = "None"

    task_meta_data_dict[task_name] = meta_data_dict

    train_data['Summary'] = task_list_dict_readme[task_name]["summary"]
    dev_data['Summary'] = task_list_dict_readme[task_name]["summary"]
    test_data['Summary'] = task_list_dict_readme[task_name]["summary"]
    task_meta_data_dict[task_name]['Summary'] = task_list_dict_readme[task_name]["summary"]

    train_data['Categories'] = source_data['Categories']
    dev_data['Categories'] = source_data['Categories']
    test_data['Categories'] = source_data['Categories']

    train_data['Definition'] = source_data['Definition']
    dev_data['Definition'] = source_data['Definition']
    test_data['Definition'] = source_data['Definition']

    train_data['Positive Examples'] = source_data['Positive Examples']
    dev_data['Positive Examples'] = source_data['Positive Examples']
    test_data['Positive Examples'] = source_data['Positive Examples']

    train_data['Negative Examples'] = source_data['Negative Examples']
    dev_data['Negative Examples'] = source_data['Negative Examples']
    test_data['Negative Examples'] = source_data['Negative Examples']

    instances = source_data["Instances"]
    random.shuffle(instances)

    train_index = int(len(instances) *  split_ratio["train"])
    dev_index = train_index +  int(len(instances) * split_ratio["dev"])
    train_data['Instances'] = source_data['Instances'][0:train_index]
    dev_data['Instances'] = source_data['Instances'][train_index:dev_index]
    test_data['Instances'] = source_data['Instances'][dev_index:]

    assert (len(instances) == len(train_data['Instances']) + len(dev_data['Instances']) +
            len(test_data['Instances']))

    return train_data, dev_data, test_data


def split_data_into_train_test_dev(input_dirpath, output_dirpath, limit_tasks=None):
    output_dirpath_train, output_dirpath_test, output_dirpath_dev = \
        create_dirs(output_dirpath)

    # ------------ get the task summaries from the readme file -----------
    with open(os.path.join(input_dirpath, "README.md"), encoding="utf-8") as fp:
        task_list_dict_readme = {}
        task_list_file_name = []
        lines = fp.readlines()
        for task in lines[7:]: # skip first 7 lines in the file
            line_items = task.split("|")
            name = line_items[0].strip().split("`")[1]
            task_list_file_name.append(name + ".json")
            summary = line_items[1].strip()
            task_list_dict_readme[name] = {"summary":summary}
            if limit_tasks and len(task_list_dict_readme) > limit_tasks:
                break

    split_ratio = {"train": 0.8, "dev": 0.1, "test": 0.1}
    total_train, total_dev, total_test = 0, 0, 0
    task_meta_data_dict = {}

    tsvfile = open(os.path.join(output_dirpath, "tasks_meta_data.tsv"), "w", encoding="utf-8")
    tsvfile.write("Task_Name\tCategories\tDomains\tInput_language\tOutput_language\tInstruction_language\tSummary\tTrain\tTest\tDev\n")

    for json_file_name in task_list_file_name:
        if not json_file_name.endswith(".json"):
            continue
        task_name = json_file_name.split(".json")[0]
        if task_name not in task_list_dict_readme:
            continue
        if not os.path.exists(os.path.join(input_dirpath, json_file_name)):
            continue
        
        with open(os.path.join(input_dirpath, json_file_name)) as json_file:
            source_data = json.load(json_file)
        train_data, dev_data, test_data = parse_task(task_list_dict_readme, split_ratio, task_meta_data_dict, task_name, source_data)

        total_train += len(train_data['Instances'])
        total_dev += len(dev_data['Instances'])
        total_test += len(test_data['Instances'])

        print("Loaded dataset [%s], train [%d/%d], dev [%d/%d], test [%d/%d]" % (
            json_file_name, len(train_data['Instances']), total_train,
            len(dev_data['Instances']), total_dev, len(test_data['Instances']), total_test))
        task_item = task_meta_data_dict[task_name]
        tsvfile.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                                    %(task_name, task_item["Categories"], task_item["Domains"], task_item["Input_language"],
                                    task_item["Output_language"], task_item["Instruction_language"], task_item["Summary"],
                                    len(train_data['Instances']), len(dev_data['Instances']), len(test_data['Instances'])))

        with open(os.path.join(output_dirpath_train, json_file_name), 'w', encoding="utf-8") as outfile:
            json.dump(train_data, outfile, indent=4)
        with open(os.path.join(output_dirpath_dev, json_file_name), 'w', encoding="utf-8") as outfile:
            json.dump(dev_data, outfile, indent=4)
        with open(os.path.join(output_dirpath_test, json_file_name), 'w', encoding="utf-8") as outfile:
            json.dump(test_data, outfile, indent=4)

    with open(os.path.join(output_dirpath, "tasks_meta_data.json"), 'w', encoding="utf-8") as outfile:
        json.dump(task_meta_data_dict, outfile, indent=4)
    tsvfile.close()


if __name__ == '__main__':
    args = parse_args()
    if args.limit_tasks < 0:
        limit_tasks = None
    else:
        limit_tasks = args.limit_tasks
    split_data_into_train_test_dev(args.input_dirpath, args.output_dirpath, limit_tasks)
