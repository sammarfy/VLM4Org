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
# class ----> Instruct_BLIP  model_name ----> "instruct-vicuna7b"                   (implemented, env: instruct_blip, --cpus-per-task=8)         #
#                                       ----> "instruct-vicuna13b"                  (implemented, env: instruct_blip, --cpus-per-task=8)         #
#                                       ----> "instruct-flant5xl"                   (implemented, env: instruct_blip, --cpus-per-task=8)         #
#                                       ----> "instruct-flant5xxl"                  (implemented, env: instruct_blip, --cpus-per-task=8)         #
##################################################################################################################################################

import json
from tqdm import tqdm
import argparse
import os

##################################################################################################################################################
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(1, parent_dir)
##################################################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default='llava-v1.5-7b', help="multimodal-model, option: 'gpt-4v', 'llava-v1.5-7b', 'llava-v1.5-13b', 'cogvlm-grounding-generalist', 'cogvlm-chat'")
parser.add_argument("--task_option", "-t", type=str, default='grounding', choices=['grounding', 'referring'], help="task option: 'grounding', 'referring' ")
parser.add_argument("--trait_option", "-r", type=str, default='head', choices=['dorsal fin','adipose fin','caudal fin','anal fin','pelvic fin','pectoral fin','head','eye'])
parser.add_argument("--result_dir", "-o", type=str, default='/projects/ml4science/project_LMM/zero_shot_SC_CoT_results/detection', help="path to output")
parser.add_argument("--num_queries", "-n", type=int, default=5, help="number of images to query from dataset")
parser.add_argument("--normalize_bbox", "-b", type=str, default='False', choices=['True','False'])
parser.add_argument("--num_runs", "-u", type=int, default=10, help="number of runs")
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

args = parser.parse_args()
args.result_dir = os.path.join(args.result_dir, args.task_option, f"Normalize_BBox_{args.normalize_bbox}")
os.makedirs(args.result_dir, exist_ok=True)

args.normalize_bbox = True if args.normalize_bbox=="True" else False

print("Arguments: ", args)

if args.model == 'gpt-4v':
    
    from interface.gpt import GPT_4V
    model = GPT_4V(model_name="gpt-4v", self_consistency = True)
    print(f'{args.model} loaded successfully.')

if args.model in ['llava-v1.5-7b', 'llava-v1.5-13b']:

    from interface.llava import LLaVA
    model_version = args.model                    
    model = LLaVA(
        model_name = model_version,
        saved_model_dir = f"/projects/ml4science/maruf/llava_models/{model_version}.pt", 
        self_consistency = True
    )

if args.model in ['cogvlm-grounding-generalist', 'cogvlm-chat']:

    from interface.cogvlm import CogVLM

    model = CogVLM(model_name=args.model, self_consistency = True)

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
from vlm_datasets.detection_dataset import DetectionDataset
import jsonlines
import json 

images_list_path = '/projects/ml4science/maruf/Fish_Data/bg_removed/metadata/sample_images.txt'
image_dir = '/projects/ml4science/maruf/Fish_Data/bg_removed/INHS'
img_metadata_path = '/projects/ml4science/maruf/Fish_Data/bg_removed/metadata/INHS.csv'
trait_map_path = '/projects/ml4science/maruf/Fish_Data/bg_removed/metadata/trait_map.pth'
segmentation_dir = '/projects/ml4science/maruf/Fish_Data/bg_removed/INHS_seg_mask/'


with open(images_list_path, 'r') as file:
    lines = file.readlines()
images_list = [line.strip() for line in lines]



detection_dataset = DetectionDataset(
                                image_dir=image_dir,
                                trait_map_path=trait_map_path,
                                segmentation_dir=segmentation_dir,
                                images_list=images_list,
                                img_metadata_path=img_metadata_path,
                                detection_type=args.task_option,
                                normalize_bbox=args.normalize_bbox,
                                )

args.num_queries = min(len(detection_dataset), args.num_queries)

out_file_name = "{}/detection_{}_{}_{}_num_{}.jsonl".format(args.result_dir, 
                                                            args.task_option,
                                                            args.model, 
                                                            args.trait_option, 
                                                            args.num_queries)

writer = jsonlines.open(out_file_name, mode='w')

#Factual Dictionary
#Prompt Used to create it: "Provide a brief one-line definition of <TRAIT>, that will be used for identifying the trait from an image."
factual_dict = {
    "adipose fin":"The adipose fin is a soft, fleshy fin found on the back behind the dorsal fin and just forward of the caudal(tail) fin.",
    "caudal fin": "The caudal fin is the tail fin of a fish, located at the end of the caudal peduncle, and is used primarily for propulsion.",
    "dorsal fin":"The dorsal fin is the fin located on the back of a fish, used for stability during swimming.",
    "pectoral fin":"Pectoral fins are the pair of fins located on either side of a fish's body just behind the head, primarily used for steering and balancing.",
    "pelvic fin":"The pelvic fins are a pair of fins located on the underside of a fish's body, near the head, used for balance and steering.",
    "anal fin":"The anal fin is a single fin located on the underside of a fish, near the tail, used for stability and maneuverability.",
    "eye":"The eye is a sensory organ located on the head of a fish, usually symmetrical on either side, used for vision.",
    "head":"The head is the anterior part of a fish, containing the brain, eyes, mouth, and gills, and is distinct from the body and fins.",
}
all_trait_desciption = " ".join(factual_dict.values())

for idx in tqdm(range(args.num_queries)):

    batch = detection_dataset[idx]

    if os.path.exists(batch['image_path']) is False:
        print(f"{batch['image_path']} does not exist!")
        continue

    if args.trait_option not in batch["present_traits"]:
        continue 

    

    target_output = batch['target_outputs'][args.trait_option]
    questions = batch['question_templates'][args.trait_option] 
    options = batch['option_templates'][args.trait_option] 
    answer_template = batch['answer_templates'][args.trait_option] 
    option_gt = batch['option_gt'][args.trait_option]

    zero_shot_CoT_prompt = "Let's think step by step."
    force_answer_prompt = "You must return an answer, which should be from the four provided options."
    
    if args.task_option == "grounding":
        factual_prompt = factual_dict[args.trait_option]
    elif args.task_option == "referring":
        factual_prompt = all_trait_desciption

    reasoning_extract_instruction = f"{factual_prompt} \n{questions} \n{options} \n{zero_shot_CoT_prompt}"

    for run in range(args.num_runs):
        result = dict()

        model_output = model.prompt(
            prompt_text= reasoning_extract_instruction,
            image_path = batch['image_path'],
        )

        if model_output is None:
            reasoning = ""
        else:
            reasoning = model_output['response']
        
        answer_extract_instruction = f"{factual_prompt} \n{questions} \n{options} \nPlease consider the following reasoning to formulate your answer: {reasoning} \n{force_answer_prompt} \nTherefore, the answer is: "

        model_output = model.prompt(
            prompt_text= answer_extract_instruction,
            image_path = batch['image_path'],
        )

        if model_output is None:
            response = "No response received."
        else:
            response = model_output['response']
        
        if args.verbose:
            print("Reasoning Extract: ",reasoning_extract_instruction)
            print("Answer Extract: ",answer_extract_instruction)
            print("Response:", response)
            print("Correct Answer: ", target_output)
            print("Correct Option: ", option_gt)
            print("\n ***************************************************************************************************************************** \n")


        result['trait'] = args.trait_option
        result['target-output'] = target_output
        result["reasoning-extract-instruction"] = reasoning_extract_instruction
        result["answer-extract-instruction"] = answer_extract_instruction
        result["reasoning"] = reasoning
        result["output"] = response
        result["run-idx"] = run
        result["image-path"] = batch['image_path']
        result["option-gt"] = option_gt

        writer.write(result)

writer.close()

