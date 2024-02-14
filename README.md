# VLM4Org

This repo contains the full pipeline and evaluation code for the paper "On the Zero-Shot Effectiveness of Pre-trained Vision-Language Models (VLMs) for Understanding Scientific Images: A Case Study in Organismal Biology"

![Alt text](assests/fig1.png)

## Abstract
With the proliferation of imaging technologies in almost every field of science, there is a growing deluge of scientific images that are being collected and made publicly accessible in a variety of disciplines. For example, in organismal biology, images are increasingly becoming the currency for documenting biodiversity on the planet, e.g., through images collected by scientists or captured by drones, camera traps, or citizen scientists.  This growth in scientific images provides a unique tipping point for accelerating discoveries in disciplines such as organismal biology that are reliant on expert visual attention to extract biological information (or traits) to understand the evolution and function of organisms.
With the advent of large foundation models such as vision-language models (VLMs) in mainstream applications of computer vision, it is pertinent to ask if pre-trained VLMs contain the necessary knowledge to aid scientists in answering biologically relevant questions without any additional fine-tuning on scientific datasets. However, unlike mainstream tasks in computer vision, understanding scientific images requires knowledge of domain-specific  terminologies  and reasoning that are not fully represented in conventional image datasets used for training VLMs. In this paper, we evaluate the effectiveness of 12 state-of-the-art (SOTA) VLMs on five scientifically relevant tasks in the field of organismal biology to study biodiversity science. To perform this evaluation, we have created a novel dataset of $~454K$ question-answer pairs based on $25k$ images of three  groups of organisms: fishes, birds, and butterflies. We also explore the effectiveness of different prompting techniques in improving the performance of VLMs on our dataset. Our analysis sheds new light on the capabilities of current standards of pre-trained VLMs in answering scientific questions involving images, prompting new research directions in this area.


## Tasks
![Alt text](assests/tasks.png)
We conducted our evaluation on five scientific tasks relevant to biologists in the study of biodiversity science. Tasks are: Species Classification, Trait Identification, Trait Grounding, Trait Referring, Trait Counting.

## Datasets
![Alt text](assests/datasets.png)
We used image collections of three taxonomic groups of organisms: Fish (contained 10k images), Bird (containing 10k images), and Butterfly (containing 5k images), obtained by taking subsets of the FishAIR dataset, the CUB dataset, and the Cambridge Butterfly dataset, respectively. The motivation for choosing these datasets is to evaluate the effectiveness of VLMs in answering biological questions over a range of bio-diverse organisms.

## Evaluation

![Alt text](assests/results.png)

## Setting up Environments

## Download Datasets

## Evaluation Notebooks

## Citation


