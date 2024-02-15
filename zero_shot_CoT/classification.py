##################################################################################################################################################
# class ----> GPT_4V         model_name ----> "gpt-4v"                                 (implemented, env: llava)                                 #
# class ----> LLaVA          model_name ----> "llava-v1.5-7b"                          (implemented, env: llava)                                 #
#                                       ----> "llava-v1.5-13b"                         (implemented, env: llava)                                 #
# class ----> OFA            model_name ----> "ofa-large"                                                                                        #
#                                       ----> "ofa-huge"                                                                                         #
# class ----> CogVLM         model_name ----> "cogvlm-grounding-generalist"            (implemented, env: vlm_env, gpu:2, --ntasks-per-node=8)   #
#                                       ----> "cogvlm-chat"                            (implemented, env: vlm_env, gpu:2, --ntasks-per-node=8)   #
# class ----> MinGPT4        model_name ----> "minigpt4-vicuna-7B"                     (implemented, env: minigptv)                              #
#                                       ----> "minigpt4-vicuna-13B"                    (implemented, env: minigptv)                              #
# class ----> BLIP-2FLAN     model_name ----> "blip-flan-xxl"                          (implemented, env: blip, --cpus-per-task=8)               #
#                                       ----> "blip-flan-xl"                           (implemented, env: blip, --cpus-per-task=8)               #
# class ----> Instruct_BLIP  model_name ----> "instruct-vicuna7b"                      (implemented, env: instruct_blip, --cpus-per-task=8)      #
#                                       ----> "instruct-vicuna13b"                     (implemented, env: instruct_blip, --cpus-per-task=8)      #
#                                       ----> "instruct-flant5xl"                      (implemented, env: instruct_blip, --cpus-per-task=8)      #
#                                       ----> "instruct-flant5xxl"                     (implemented, env: instruct_blip, --cpus-per-task=8)      #
##################################################################################################################################################

import json
from tqdm import tqdm
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

##################################################################################################################################################
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(1, parent_dir)
##################################################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default='llava-v1.5-7b', help="multimodal-model, option: 'gpt-4v', 'llava-v1.5-7b', 'llava-v1.5-13b', 'cogvlm-grounding-generalist', 'cogvlm-chat'")
parser.add_argument("--task_option", "-t", type=str, default='direct', help="task option: 'direct', 'selection' ")
parser.add_argument("--num_queries", "-n", type=int, default=-1, help="number of images to query from dataset")
parser.add_argument("--chunk_id", "-c", type=int, default=0, help="0, 1, 2, 3, 4, 5, 6, 7, 8, 9")

# updated
parser.add_argument("--dataset", "-d", type=str, default='fish-500', help="dataset option: 'fish-10k', 'fish-500', 'bird', 'butterfly' ")
parser.add_argument("--server", "-s", type=str, default='arc', help="server option: 'osc', 'pda'")

args = parser.parse_args()

if args.server == "arc":
    root_dir = '/projects/ml4science/'
elif args.server == "osc":
    root_dir = '/fs/ess/PAS2136/'
elif args.server == "pda":
    root_dir = '/data/'


if args.dataset == 'fish-10k':

    args.result_dir = root_dir+'VLM4Bio/zero_shot_CoT_results/fish-10k'
    
    images_list_path = root_dir+'VLM4Bio/datasets/Fish/metadata/imagelist_10k.txt'
    image_dir = root_dir+'VLM4Bio/datasets/Fish/images'
    img_metadata_path = root_dir+'VLM4Bio/datasets/Fish/metadata/metadata_10k.csv'
    organism = 'fish'

elif args.dataset == 'fish-500':

    args.result_dir = root_dir+'VLM4Bio/zero_shot_CoT_results/fish-500'
    
    images_list_path = root_dir+'VLM4Bio/datasets/Fish/metadata/imagelist_500.txt'
    image_dir = root_dir+'VLM4Bio/datasets/Fish/images'
    img_metadata_path = root_dir+'VLM4Bio/datasets/Fish/metadata/metadata_500.csv'
    organism = 'fish'

elif args.dataset == 'bird':

    args.result_dir = root_dir+'VLM4Bio/zero_shot_CoT_results/bird'
    
    images_list_path = root_dir+'VLM4Bio/datasets/Bird/metadata/bird_imagelist_10k.txt'
    image_dir = root_dir+'VLM4Bio/datasets/Bird/images'
    img_metadata_path = root_dir+'VLM4Bio/datasets/Bird/metadata/bird_metadata_10k.csv'
    organism = 'bird'

elif args.dataset == 'butterfly':

    args.result_dir = root_dir+'VLM4Bio/zero_shot_CoT_results/butterfly'
    
    images_list_path = root_dir+'VLM4Bio/datasets/Butterfly/metadata/imagelist.txt'
    image_dir = root_dir+'VLM4Bio/datasets/Butterfly/images'
    img_metadata_path = root_dir+'VLM4Bio/datasets/Butterfly/metadata/metadata.csv'
    organism = 'butterfly'



args.result_dir = os.path.join(args.result_dir, 'classification' , args.task_option)

os.makedirs(args.result_dir, exist_ok=True)


print("Arguments: ", args)

if args.model == 'gpt-4v':
    
    from interface.gpt import GPT_4V
    model = GPT_4V(model_name="gpt-4v")
    print(f'{args.model} loaded successfully.')

if args.model in ['llava-v1.5-7b', 'llava-v1.5-13b']:

    from interface.llava import LLaVA
    model_version = args.model                    
    model = LLaVA(
        model_name = model_version,
        saved_model_dir = root_dir+f"VLM4Bio/llava_models/{model_version}.pt"
    )

if args.model in ['cogvlm-grounding-generalist', 'cogvlm-chat']:

    from interface.cogvlm import CogVLM

    model = CogVLM(model_name=args.model)

if args.model in ['minigpt4-vicuna-7B', 'minigpt4-vicuna-13B']:
    
    from interface.minigpt4 import MiniGPT4

    model = MiniGPT4(model_name=args.model,
        cfg_path=f'minigpt4_eval_configs/eval_{args.model}.yaml',
        model_cfg_name=f'{args.model}.yaml'
    )

if args.model in ['blip-flan-xxl', 'blip-flan-xl']:
    
    from interface.blip import BLIP

    model = BLIP(model_name=args.model)

if args.model in ['instruct-vicuna7b', 'instruct-vicuna13b', 'instruct-flant5xl', 'instruct-flant5xxl']:
    
    from interface.instruct_blip import Instruct_BLIP

    model = Instruct_BLIP(model_name=args.model)

##########################################################################################################################
from vlm_datasets.species_dataset import SpeciesClassificationDataset
import jsonlines
import json 

# images_list_path = '/projects/ml4science/maruf/Fish_Data/bg_removed/metadata/sample_images.txt'
# image_dir = '/projects/ml4science/maruf/Fish_Data/bg_removed/INHS'
# img_metadata_path = '/projects/ml4science/maruf/Fish_Data/bg_removed/metadata/INHS.csv'

with open(images_list_path, 'r') as file:
    lines = file.readlines()
images_list = [line.strip() for line in lines]

chunk_len = len(images_list)//10
start_idx = chunk_len * args.chunk_id
end_idx = len(images_list) if args.chunk_id == 9 else (chunk_len * (args.chunk_id+1))
images_list = images_list[start_idx:end_idx]
args.num_queries = len(images_list) if args.num_queries == -1 else args.num_queries


species_dataset = SpeciesClassificationDataset(images_list=images_list, 
                                               image_dir=image_dir, 
                                               img_metadata_path=img_metadata_path)

args.num_queries = min(len(species_dataset), args.num_queries)


out_file_name = "{}/classification_{}_{}_num_{}_chunk_{}.jsonl".format(args.result_dir, args.model, args.task_option, args.num_queries, args.chunk_id)


if os.path.exists(out_file_name):

    print('Existing result file found!')
    queried_files = []

    # read the files that has been already written
    with open(out_file_name, 'r') as file:
        # Iterate over each line
        for line in file:
            # Parse the JSON data
            data = json.loads(line)
            queried_files.append(data['image-path'].split('/')[-1])


    images_list = list(set(images_list) - set(queried_files))
    print(f'Running on the remaining {len(images_list)} files.')

    species_dataset = SpeciesClassificationDataset(images_list=images_list, 
                                               image_dir=image_dir, 
                                               img_metadata_path=img_metadata_path)

    args.num_queries = min(len(species_dataset), args.num_queries)


    writer = jsonlines.open(out_file_name, mode='a')

else:
    writer = jsonlines.open(out_file_name, mode='w')


for idx in tqdm(range(args.num_queries)):

    batch = species_dataset[idx]

    if os.path.exists(batch['image_path']) is False:
        print(f"{batch['image_path']} does not exist!")
        continue

    result = dict()

    target_species = batch['target_outputs'][args.task_option]
    questions = batch['question_templates'][args.task_option] 
    options = batch['option_templates'][args.task_option] 
    answer_template = batch['answer_templates'][args.task_option] 
    
    zero_shot_CoT_prompt = "Let's think step by step"
    if args.task_option == "direct":
        force_answer_prompt = "You must return an answer. Always return the most plausible answer."
    elif args.task_option == "selection":
        force_answer_prompt = "You must return an answer, which should be from the four provided options."

	# confused_prompt = 'If the information is not enough to answer the question, answer \"lack of information\"'

    questions = "Each biological species has a unique scientific name composed of two parts: the first for the genus and the second for the species within that genus. \nWhat is the scientific name of the fish in the image?"
    reasoning_extract_instruction = f"{questions} \n{options} \n{zero_shot_CoT_prompt}."
    reasoning_extract_instruction = reasoning_extract_instruction.replace('fish', organism)

    model_output = model.prompt(
		prompt_text= reasoning_extract_instruction,
		image_path = batch['image_path'],
	)

    if model_output is None:
        reasoning = ""
    else:
        reasoning = model_output['response']

    answer_extract_instruction = f"{questions} \n{options} \nPlease consider the following reasoning to formulate your answer: {reasoning} \n{force_answer_prompt} \nTherefore, the answer is: "

    model_output = model.prompt(
		prompt_text= answer_extract_instruction,
		image_path = batch['image_path'],
	)

    if model_output is None:
        response = "No response received."
    else:
        response = model_output['response']
    
    result['question'] = answer_extract_instruction
    result['target-class'] = target_species

    if model_output is None:
        response = "No response received."
    else:
        response = model_output['response']
		
    result["reasoning-extract-instruction"] = reasoning_extract_instruction
    result["answer-extract-instruction"] = answer_extract_instruction
    result["reasoning"] = reasoning
        
    result["output"] = response

    result["image-path"] = batch['image_path']
    result["option-gt"] = batch['option_gt'][args.task_option]
    writer.write(result)
    writer.close()
    writer = jsonlines.open(out_file_name, mode='a')
writer.close()

