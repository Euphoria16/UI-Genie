import json
import re
import os
from collections import defaultdict
import math
from PIL import Image, ImageDraw, ImageFont
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import smart_resize

def read_jsonl(meta_path):
    if meta_path.endswith('.jsonl'):
        meta_l = []
        with open(meta_path) as f:
            for i, line in enumerate(f):
                try:
                    meta_l.append(json.loads(line))
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}", force=True)
                    raise e
    else:
        with open(meta_path, 'r') as f1:
            meta_l = json.load(f1)
    return meta_l

def parse_action(value_str):
    
    try:
        if isinstance(value_str, str):
            value_str = value_str.strip()
            if '<tool_call>\n' in value_str:
                
                value_str = value_str.split('<tool_call>\n')[1].split('\n</tool_call>')[0]
            
            action = json.loads(value_str)
        else:
            action=value_str
        action = action['arguments']
        return action
    except Exception as e:
        print('error parsing action', value_str)
        match = re.search(r'(\{.*\})', value_str)
        if match:
            action_str = match.group(1)
            try:
                action = json.loads(action_str)
                return action
            except json.JSONDecodeError:
                return None
        else:
            return None


    
def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def determine_swipe_direction(start, end):
    try:
        x0, y0 = start
        x1, y1 = end
    except Exception as e:
        
        return "right"
    delta_x = x1 - x0
    delta_y = y1 - y0

    if abs(delta_x) > abs(delta_y):
        if delta_x > 0:
            return "right"
        else:
            return "left"
    else:
        if delta_y > 0:
            return "down"
        else:
            return "up"
def compare_actions(answer_action, gpt_action, raw_id, reverse_swipe=True, width=None, height = None): 
    type_equal = False
    if not answer_action or not gpt_action:
        return False, False
    answer_action_type = answer_action.get('action', '').lower()
    gpt_action_type = gpt_action.get('action', '').lower()
    if answer_action_type != gpt_action_type:
        return False, type_equal

    type_equal = True
    # if 'coordinate' in gpt_action and ('coordinate2' not in gpt_action) and (gpt_action_type!='swipe'):
    if ('click' in answer_action_type) or ('long_press' in answer_action_type):
        x, y = gpt_action['coordinate']
        
        resized_height, resized_width  = smart_resize(height,
            width,
            factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels=processor.image_processor.min_pixels,
            max_pixels=processor.image_processor.max_pixels,)
        x = int(float(x/resized_width*width))
        y = int(float( y/resized_height*height))


        answer_xy = answer_action.get('coordinate')

        if answer_xy is None:
            answer_x = 0
            answer_y = 0
        else:
            answer_x , answer_y = answer_xy

            
        answer_x = int(float(answer_x/resized_width*width))
        answer_y = int(float( answer_y/resized_height*height))
        try:
            answer_x = int(float(answer_x))
            answer_y = int(float(answer_y))
        except (TypeError, ValueError):
            return False, type_equal

        if math.sqrt((answer_x - x)**2 + (answer_y - y)**2) <= math.sqrt((width*0.14)**2 + (height*0.14)**2):
            return True, type_equal

        else:
            return False, type_equal

    else:
        if answer_action_type == 'type':
            answer_text = answer_action.get('text', '').strip().lower()
            gpt_text = gpt_action.get('text', '').strip().lower()
            answer_text = re.sub(r'\s+', ' ', answer_text)
            gpt_text = re.sub(r'\s+', ' ', gpt_text)
            if gpt_text in answer_text or answer_text in gpt_text:
                return True, type_equal
            else:
                # f1_score = calculate_f1_score(gpt_text, answer_text)
                # if f1_score >0.5:
                #     return True, selected_bound
            

                # else:
                return False, type_equal
        elif answer_action_type == 'swipe':
            if 'direction' in answer_action:
                answer_direction = answer_action.get('direction', '')
                
            else:
                start_point = answer_action.get('coordinate', '')
                end_point = answer_action.get('coordinate2', '')
                if not isinstance(start_point, list):
                    # print('start_point', start_point)
                    answer_direction=end_point
                else:
                    
                    answer_direction = determine_swipe_direction(start_point, end_point)
            gpt_direction = gpt_action.get('direction', '').strip().lower() or gpt_action.get('coordinate', '').strip().lower()
            if reverse_swipe:
                if 'up' in gpt_direction:
                    gpt_direction ='down'
                elif 'down' in gpt_direction:
                    gpt_direction='up'
                elif 'left' in gpt_direction:
                    gpt_direction='right'
                elif 'right' in gpt_direction:
                    gpt_direction='left'
            if gpt_direction in answer_direction or answer_direction in gpt_direction:
                return True, type_equal
            else:
                return False, type_equal
        elif answer_action_type == 'open':
            
            answer_app_name = answer_action.get('text', '').strip().lower()
            gpt_app_name = gpt_action.get('text', '').strip().lower()
            if gpt_app_name in answer_app_name or answer_app_name in gpt_app_name or (calculate_f1_score(answer_app_name, gpt_app_name)>0.5):
                return True, type_equal
            else:

                return False, type_equal
        elif answer_action_type == 'terminate':
            answer_goal_status = answer_action.get('status', '').strip().lower()
            gpt_goal_status = gpt_action.get('status', '').strip().lower()
            if gpt_goal_status in answer_goal_status or answer_goal_status in gpt_goal_status or (calculate_f1_score(answer_goal_status, gpt_goal_status)>0.5):
                return True, type_equal
            else:

                return False, type_equal
        elif answer_action_type == 'system_button':
            answer_button = answer_action.get('button', '').strip().lower()
            gpt_button = gpt_action.get('button', '').strip().lower()
            if gpt_button in answer_button or answer_button in gpt_button:
                return True, type_equal
            else:
                return False, type_equal
        elif answer_action_type in [ 'wait']:
            return True, type_equal
        else:
            print('unrecognized action', answer_action_type, answer_action, gpt_action)
            return False, type_equal

def process_data(gt_jsonl_data, answer_json_data,save_name, revise_swipe=True, save_dir=''):

    

    action_types = {"system_button", "click", "terminate", "swipe", "type", "wait", "long_press" , "open" }
    action_error_data = { action_type: {
        'all_count': 0,
        'error_count': 0,
        'actiontype_notmatch_numbers': 0,
        'actiontype_notmatch_details': [],
        'actiontype_match_numbers': 0,
        'actiontype_match_details': []
    } for action_type in action_types }

    
    
    all_actions = 0
    all_grounding_count = 0
    action_counts = defaultdict(int)
    correct_action_counts = defaultdict(int)
    type_correct_action_counts = defaultdict(int)
    grounding_correct_action_counts = defaultdict(int)
    task_subtask_counts = defaultdict(int)
    task_correct_subtask_counts = defaultdict(int)
    ep_correctness = {}
    difficulty_levels = {}
    correct_questionIds = set()

    details = defaultdict(list)
    error_examples = defaultdict(list)
    click_error_examples = []
    skipped_questions = []



    for data_item in gt_jsonl_data:
        raw_id = os.path.basename(data_item['images'][0]).split('.')[0]
        task_id = raw_id.split('_')[0]
        task_subtask_counts[task_id] += 1


    for  item, pred_item in zip(gt_jsonl_data, answer_json_data):
        raw_id = os.path.basename(item['images'][0]).split('.')[0]
        assert os.path.basename(item['images'][0]).split('_')[0]==os.path.basename(pred_item['images'][0]['path']).split('_')[0]
        img_dir = item['images'][0]

        gt_action = None
        gt_value_str = ''
        for msg in item['messages']:
            if msg['role']=='assistant':
                gt_value_str = msg['content']
        gt_action  = parse_action(gt_value_str)


        if not gt_action:
            skipped_questions.append(f"'{raw_id}, reason: missing gpt_action")
            continue

        action_type = gt_action.get('action', '').lower()
        if not action_type:
            skipped_questions.append(f"'{raw_id}, reason: missing action_type in gpt_action")
            continue



        if action_type in action_types:
            action_error_data[action_type]['all_count'] += 1

    
        answer_action  = parse_action(pred_item['response'])
        if not answer_action:
            continue

        
        w,h = item['image_wh']
        is_correct, type_equal = compare_actions(answer_action, gt_action, raw_id, reverse_swipe=revise_swipe, width=w, height=h)
        answer_action_type = answer_action.get('action', '').lower()
        all_actions += 1
        action_counts[action_type] += 1
        if ('coordinate' in gt_action and ('coordinate' in answer_action)):
            all_grounding_count +=1
            if is_correct:
                grounding_correct_action_counts[action_type] += 1
            

        if type_equal:
            type_correct_action_counts[action_type] += 1
        if is_correct:
            
            correct_action_counts[action_type] += 1
            correct_questionIds.add(raw_id)
            details_key = f"{action_type}_success_action"
            task_id = raw_id.split('_')[0]
            task_correct_subtask_counts[task_id] += 1
        else:
            details_key = f"{action_type}_error_action"
            if len(error_examples[details_key]) < 100:
                error_examples[details_key].append({
                    "questionID": raw_id,
                    "answer": json.dumps(answer_action, ensure_ascii=False),
                    "gt": gt_value_str
                })
            if action_type == 'click' and len(click_error_examples) < 50:
                click_error_examples.append({
                    "questionId": raw_id,
                    "answer": answer_action,
                    "gpt_action": gt_action,
                    "selected_bound": type_equal
                })

            if action_type in action_types:
                action_error_data[action_type]['error_count'] += 1
                if answer_action_type == action_type:
                    action_error_data[action_type]['actiontype_match_numbers'] += 1
                    action_error_data[action_type]['actiontype_match_details'].append({
                        "questionID": raw_id,
                        "gt": gt_value_str,
                        "answer": json.dumps(answer_action, ensure_ascii=False)
                    })
                else:
                    action_error_data[action_type]['actiontype_notmatch_numbers'] += 1
                    action_error_data[action_type]['actiontype_notmatch_details'].append({
                        "questionID": raw_id,
                        "gt": gt_value_str,
                        "answer": json.dumps(answer_action, ensure_ascii=False)
                    })

        details[details_key].append(raw_id)

    all_ep = len(task_subtask_counts)
    ep_correctness = {}
    for task_id, total_subtasks in task_subtask_counts.items():
        correct_subtasks = task_correct_subtask_counts.get(task_id, 0)
        if total_subtasks == correct_subtasks:
            ep_correctness[task_id] = True
        else:
            ep_correctness[task_id] = False

    correct_eps = sum(1 for correct in ep_correctness.values() if correct)   


    for task_id, subtask_count in task_subtask_counts.items():
        if 1 <= subtask_count < 5:
            level = 'easy'
        elif 5 <= subtask_count < 10:
            level = 'medium'
        else:
            level = 'hard'
        difficulty_levels[task_id] = level

    level_action_counts = defaultdict(int)
    level_correct_action_counts = defaultdict(int)
    level_ep_counts = defaultdict(int)
    level_correct_ep_counts = defaultdict(int)

    for task_id, level in difficulty_levels.items():
        level_ep_counts[level] += 1
        if ep_correctness[task_id]:
            level_correct_ep_counts[level] += 1
        total_subtasks = task_subtask_counts[task_id]
        correct_subtasks = task_correct_subtask_counts.get(task_id, 0)
        level_action_counts[level] += total_subtasks
        level_correct_action_counts[level] += correct_subtasks

    action_acc = sum(correct_action_counts.values()) / all_actions if all_actions > 0 else 0
    type_action_acc = sum(type_correct_action_counts.values()) / all_actions if all_actions > 0 else 0
    grounding_action_acc = sum(grounding_correct_action_counts.values()) / all_grounding_count if all_grounding_count > 0 else 0
    ep_acc = correct_eps / all_ep if all_ep > 0 else 0

    action_acc_easy = (level_correct_action_counts.get('easy', 0) / level_action_counts.get('easy', 0)) if level_action_counts.get('easy', 0) > 0 else 0
    ep_acc_easy = (level_correct_ep_counts.get('easy', 0) / level_ep_counts.get('easy', 0)) if level_ep_counts.get('easy', 0) > 0 else 0

    action_acc_medium = (level_correct_action_counts.get('medium', 0) / level_action_counts.get('medium', 0)) if level_action_counts.get('medium', 0) > 0 else 0
    ep_acc_medium = (level_correct_ep_counts.get('medium', 0) / level_ep_counts.get('medium', 0)) if level_ep_counts.get('medium', 0) > 0 else 0

    action_acc_hard = (level_correct_action_counts.get('hard', 0) / level_action_counts.get('hard', 0)) if level_action_counts.get('hard', 0) > 0 else 0
    ep_acc_hard = (level_correct_ep_counts.get('hard', 0) / level_ep_counts.get('hard', 0)) if level_ep_counts.get('hard', 0) > 0 else 0

    def calculate_action_type_acc(action_type):
        total = action_counts.get(action_type, 0)
        correct = correct_action_counts.get(action_type, 0)
        return (correct / total) if total > 0 else 0

    type_acc = calculate_action_type_acc('type')
    swipe_acc = calculate_action_type_acc('swipe')
    click_acc = calculate_action_type_acc('click')
    open_acc = calculate_action_type_acc('open')
    terminate_acc = calculate_action_type_acc('terminate')
    system_button_acc = calculate_action_type_acc('system_button')
    wait_acc = calculate_action_type_acc('wait')
    long_press_acc = calculate_action_type_acc('long_press')

    # Summarize data
    acc_data = {
        "action_acc": action_acc,
        "type_match_acc": type_action_acc,
        "grounding_acc": grounding_action_acc,
        "ep_acc": ep_acc,
        "all_actions": all_actions,
        "all_ep": all_ep,
        "all_click": action_counts.get('click', 0),
        "all_swipe": action_counts.get('swipe', 0),
        "all_type": action_counts.get('type', 0),
        "all_open": action_counts.get('open', 0),
        "all_terminate": action_counts.get('terminate', 0),
        "all_system_button": action_counts.get('system_button', 0),
        "all_wait": action_counts.get('wait', 0),
        "all_long_press": action_counts.get('long_press', 0),
        "action_acc_easy": action_acc_easy,
        "ep_acc_easy": ep_acc_easy,
        "action_acc_medium": action_acc_medium,
        "ep_acc_medium": ep_acc_medium,
        "action_acc_hard": action_acc_hard,
        "ep_acc_hard": ep_acc_hard,
        "type_acc": type_acc,
        "swipe_acc": swipe_acc,
        "click_acc": click_acc,
        "open_acc": open_acc,
        "terminate_acc": terminate_acc,
        "system_button_acc": system_button_acc,
        "wait_acc": wait_acc,
        "long_press_acc": long_press_acc,
    }

    # Save details.json
    required_details_keys = [
        'click_success_action', 'click_error_action',
        'type_success_action', 'type_error_action',
        'open_app_success_action', 'open_app_error_action',
        'complete_success_action', 'complete_error_action',
        'navigate_home_success_action', 'navigate_home_error_action',
        'navigate_back_success_action', 'navigate_back_error_action',
        'wait_success_action', 'wait_error_action',
        'long_press_success_action', 'long_press_error_action',
        'enter_success_action', 'enter_error_action',
        'scroll_success_action', 'scroll_error_action'
    ]

    for key in required_details_keys:
        if key not in details:
            details[key] = []

    example_list = []
    for error_type, examples in error_examples.items():
        example_list.extend(examples)

    save_json_file(example_list, f'{save_dir}/example.json')   


    for action_type in action_types:
        badcase_data = {
            f"error_{action_type}": action_error_data[action_type]['error_count'],
            f"all_{action_type}": action_error_data[action_type]['all_count'],
            "actiontype_notmatch_gt": {
                "actiontype_notmatch_numbers": action_error_data[action_type]['actiontype_notmatch_numbers'],
                "actiontype_notmatch_details": action_error_data[action_type]['actiontype_notmatch_details']
            },
            "actiontype_match_gt": {
                "actiontype_match_numbers": action_error_data[action_type]['actiontype_match_numbers'],
                "actiontype_match_details": action_error_data[action_type]['actiontype_match_details']
            }
        }
        badcase_file_path = os.path.join(save_dir, f"{action_type}_badcase.json")
        save_json_file(badcase_data, badcase_file_path)

    return acc_data, details

      
def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main(input_name, pred_name, revise_swipe=True):
    save_name = os.path.basename(pred_name).split('.json')[0]
    save_dir = f'androidcontrol_infer_acc/{save_name}'
    os.makedirs(save_dir, exist_ok=True)            


    gt_data = read_jsonl(input_name)
    pred_data = read_jsonl(pred_name)
    acc_data, details = process_data(gt_data, pred_data, save_name, revise_swipe, save_dir)
    print('Save accuracy to ', f'{save_dir}/acc.json')
    save_json_file(acc_data, f'{save_dir}/acc.json')
    save_json_file(details, f'{save_dir}/details.json')
    print('accuracy:\n', acc_data)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--input_name', type=str, help='The name to save the results under')

    parser.add_argument('--pred_name', type=str, help='The name to save the results under')
    parser.add_argument('--model_path', type=str, help='The name to save the results under')
    parser.add_argument('--max_pixels', type=int, help='The name to save the results under')
    parser.add_argument('--reverse_swipe', action='store_true', default=False, help='The name to save the results under')
    
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)

    main(args.input_name, args.pred_name, args.reverse_swipe)

