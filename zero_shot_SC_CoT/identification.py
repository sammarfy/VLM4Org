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
import pdb
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
parser.add_argument("--num_runs", "-u", type=int, default=3, help="number of runs")
parser.add_argument("--chunk_id", "-c", type=int, default=0, help="0, 1, 2, 3, 4, 5, 6, 7, 8, 9")

# updated
parser.add_argument("--dataset", "-d", type=str, default='bird', help="dataset option: 'fish-10k','bird' ")
parser.add_argument("--server", "-s", type=str, default='arc', help="server option: 'osc', 'pda'")
args = parser.parse_args()

if args.server == "arc":
    root_dir = '/projects/ml4science/'
elif args.server == "osc":
    root_dir = '/fs/ess/PAS2136/'
elif args.server == "pda":
    root_dir = '/data/'

if args.dataset == 'fish-10k':

    args.result_dir = root_dir+'VLM4Bio/zero_shot_SC_CoT_results/fish-10k'
    
    images_list_path = root_dir+'VLM4Bio/datasets/Fish/metadata/processed_identification_imagelist_10k.txt'
    image_dir = root_dir+'VLM4Bio/datasets/Fish/images'
    img_metadata_path = root_dir+'VLM4Bio/datasets/Fish/metadata/metadata_10k.csv'
    identification_metadata_path = root_dir+'VLM4Bio/datasets/Fish/metadata/processed_identification_matrix.csv'
    organism = 'fish'


elif args.dataset == 'bird':

    args.result_dir = root_dir+'VLM4Bio/zero_shot_SC_CoT_results/bird'
    
    images_list_path = root_dir+'VLM4Bio/datasets/Bird/metadata/bird_imagelist_10k.txt'
    image_dir = root_dir+'VLM4Bio/datasets/Bird/images'
    img_metadata_path = root_dir+'VLM4Bio/datasets/Bird/metadata/bird_metadata_10k.csv'
    identification_metadata_path = root_dir+'VLM4Bio/datasets/Bird/metadata/processed_identification.csv'
    trait_category_map_path = root_dir+'VLM4Bio/datasets/Bird/metadata/trait_category_map.pkl'
    organism = 'bird'


args.result_dir = os.path.join(args.result_dir, 'identification' ,args.task_option)

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
from vlm_datasets.identification_dataset import BirdIdentificationDataset
from vlm_datasets.identification_dataset import FishIdentificationDataset
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

#Factual Dictionary
#Prompt Used to create it: "Provide a brief one-line definition of <TRAIT>, that will be used for identifying the trait from an image."
#'eye', 'head', 'mouth', 'barbel', 'dorsal fin', 'two dorsal fins', 'adipose fin', 'pectoral fin', 'pelvic fin', 'anal fin'
factual_dict_fish = {
    "adipose fin":"The adipose fin is a soft, fleshy fin found on the back behind the dorsal fin and just forward of the caudal(tail) fin.",
    "caudal fin": "The caudal fin is the tail fin of a fish, located at the end of the caudal peduncle, and is used primarily for propulsion.",
    "dorsal fin":"The dorsal fin is the fin located on the back of a fish, used for stability during swimming.",
    "two dorsal fins":"The dorsal fin is the fin located on the back of a fish, used for stability during swimming. There can be two dorsal fins present for some fish species.",
    "pectoral fin":"Pectoral fins are the pair of fins located on either side of a fish's body just behind the head, primarily used for steering and balancing.",
    "pelvic fin":"The pelvic fins are a pair of fins located on the underside of a fish's body, near the head, used for balance and steering.",
    "anal fin":"The anal fin is a single fin located on the underside of a fish, near the tail, used for stability and maneuverability.",
    "eye":"The eye is a sensory organ located on the head of a fish, usually symmetrical on either side, used for vision.",
    "head":"The head is the anterior part of a fish, containing the brain, eyes, mouth, and gills, and is distinct from the body and fins.",
    "mouth":"The mouth of a fish is the opening through which it ingests food and breathes, typically located on the front part of its head.",
    "barbel":"A barbel of a fish is a fleshy, whisker-like projection usually found around the mouth or head area, often used for sensing the environment and locating food.",
}
# ['back-color', 'back-pattern', 'belly-color', 'belly-pattern', 'bill-color', 'bill-length', 'bill-shape', 'breast-color', 'breast-pattern', 'crown-color', 'eye-color', 'forehead-color', 'head-pattern', 'leg-color', 'nape-color', 'primary-color', 'shape', 'size', 'tail-pattern', 'tail-shape', 'throat-color', 'under-tail-color', 'underparts-color', 'upper-tail-color', 'upperparts-color', 'wing-color', 'wing-pattern', 'wing-shape']
factual_dict_bird = {
    'back-color': "The back color refers to the hue or shade on the uppermost part of a bird's body, typically its dorsal side.",
    'back-pattern': "Back pattern describes the design or markings present on the upper part of a bird's body.",
    'belly-color': "Belly color pertains to the shade or coloration found on the underside or ventral side of a bird.",
    'belly-pattern': "Belly pattern represents the distinctive markings or designs on the lower part of a bird's body.",
    'bill-color': "Bill color is the hue or shade of a bird's beak or bill.",
    'bill-length': "Bill length measures the extent or size of a bird's beak, from the base to the tip.",
    'bill-shape': "Bill shape characterizes the form or structure of a bird's beak, such as pointed, hooked, or conical.",
    'breast-color': "Breast color refers to the coloration on the front portion of a bird's body, typically its chest.",
    'breast-pattern': "Breast pattern represents the unique markings or designs on the front of a bird's body.",
    'crown-color': "Crown color is the hue or shade on the top of a bird's head.",
    'eye-color': "Eye color indicates the coloration of a bird's eyes, usually located on the sides of the head.",
    'forehead-color': "Forehead color pertains to the hue or shade on the front part of a bird's head, just above the beak.",
    'head-pattern': "Head pattern describes the markings or designs present on the head of a bird.",
    'leg-color': "Leg color is the coloration of a bird's legs or lower limb area.",
    'nape-color': "Nape color refers to the hue or shade on the back of a bird's neck.",
    'primary-color': "Primary color represents the dominant hue of a bird's feathers, often found on its wings.",
    'shape': "Shape relates to the overall form or silhouette of a bird, including its body structure and posture.",
    'size': "Size indicates the physical dimensions of a bird, such as its length and wingspan.",
    'tail-pattern': "Tail pattern characterizes the markings or designs on a bird's tail feathers.",
    'tail-shape': "Tail shape describes the form or structure of a bird's tail, whether forked, pointed, or rounded.",
    'throat-color': "Throat color pertains to the hue or shade on the front part of a bird's neck.",
    'under-tail-color': "Under-tail color refers to the coloration on the underside of a bird's tail feathers.",
    'underparts-color': "Underparts color represents the shade or coloration on the lower part of a bird's body.",
    'upper-tail-color': "Upper-tail color is the hue or shade on the upper side of a bird's tail feathers.",
    'upperparts-color': "Upperparts color pertains to the coloration on the uppermost part of a bird's body.",
    'wing-color': "Wing color refers to the hue or shade on a bird's wings.",
    'wing-pattern': "Wing pattern describes the markings or designs present on a bird's wings.",
    'wing-shape': "Wing shape characterizes the form or structure of a bird's wings, including their size and shape."
}



if args.dataset == "bird":
    identification_dataset = BirdIdentificationDataset(images_list=images_list, 
                                                image_dir=image_dir, 
                                                img_metadata_path=img_metadata_path,
                                                identification_metadata_path = identification_metadata_path,
                                                trait_category_map_path=trait_category_map_path
                                                )
    factual_dict = factual_dict_bird

elif args.dataset == "fish-10k":
    identification_dataset = FishIdentificationDataset(images_list=images_list, 
                                                image_dir=image_dir, 
                                                img_metadata_path=img_metadata_path,
                                                identification_metadata_path = identification_metadata_path
                                                )
    factual_dict = factual_dict_fish

args.num_queries = min(len(identification_dataset), args.num_queries)

out_file_name = "{}/identification_{}_{}_num_{}_chunk_{}.jsonl".format(args.result_dir, args.model, args.task_option, args.num_queries, args.chunk_id)



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

    if args.dataset == "bird":
        identification_dataset = BirdIdentificationDataset(images_list=images_list, 
                                                    image_dir=image_dir, 
                                                    img_metadata_path=img_metadata_path,
                                                    identification_metadata_path = identification_metadata_path,
                                                    trait_category_map_path=trait_category_map_path
                                                    )
    elif args.dataset == "fish-10k":
        identification_dataset = FishIdentificationDataset(images_list=images_list, 
                                                    image_dir=image_dir, 
                                                    img_metadata_path=img_metadata_path,
                                                    identification_metadata_path = identification_metadata_path
                                                    )
    args.num_queries = min(len(identification_dataset), args.num_queries)


    writer = jsonlines.open(out_file_name, mode='a')

else:
    writer = jsonlines.open(out_file_name, mode='w')



for idx in tqdm(range(args.num_queries)):

    batch = identification_dataset[idx]

    if os.path.exists(batch['image_path']) is False:
        print(f"{batch['image_path']} does not exist!")
        continue

    for trait in batch['unique_traits']:
        result = dict()
        factual_desc = factual_dict[trait]
        questions = batch['question_templates'][trait]
        options = batch['option_templates'][trait]
        answer_temp = {batch['answer_templates'][trait]}
        
        zero_shot_CoT_prompt = "Let's think step by step."
        reasoning_extract_instruction = f"{factual_desc} \n{questions} \n{options} \n{zero_shot_CoT_prompt}"
        for run in range(args.num_runs):

            model_output = model.prompt(
                prompt_text= reasoning_extract_instruction,
                image_path = batch['image_path'],
            )

            if model_output is None:
                reasoning = ""
            else:
                reasoning = model_output['response']

            answer_extract_instruction = f"{factual_desc} \n{questions} \n{options} \nPlease consider the following reasoning to formulate your answer: {reasoning} \nTherefore, the answer is: "
            
            model_output = model.prompt(
                prompt_text= answer_extract_instruction,
                image_path = batch['image_path'],
            )

            if model_output is None:
                response = "No response received."
            else:
                response = model_output['response']
        

            target_id = f"{batch['target_outputs'][trait]}"

            result['trait'] = trait
            result['target-identification'] = target_id
            result['target-option_id'] = batch['option_gt'][trait]
            result["reasoning-extract-instruction"] = reasoning_extract_instruction
            result["answer-extract-instruction"] = answer_extract_instruction
            result["reasoning"] = reasoning
            result["output"] = response
            result["run-idx"] = run

            result["image-path"] = batch['image_path']
            writer.close()
        writer = jsonlines.open(out_file_name, mode='a')

        writer.write(result)
        writer.close()
        writer = jsonlines.open(out_file_name, mode='a')
    writer.close()
    writer = jsonlines.open(out_file_name, mode='a')
writer.close()
