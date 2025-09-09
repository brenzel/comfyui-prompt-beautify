import os
from llama_cpp import Llama
import folder_paths
import requests

llm_dir = os.path.join(folder_paths.models_dir, "LLM")
if not os.path.exists(llm_dir):
    os.makedirs(llm_dir)


class PromptBeautify:

    @classmethod
    def INPUT_TYPES(s):

        llm_names = []
        for file in os.listdir(llm_dir):
            if file.lower().endswith("gguf"):
                llm_names.append(file)
        
        #QWEN
        if "Qwen3-0.6B-Q4_K_M.gguf" not in llm_names:
            llm_names.append("Qwen3-0.6B-Q4_K_M.gguf")
        
        if "Qwen3-1.7B-Q4_K_M.gguf" not in llm_names:
            llm_names.append("Qwen3-1.7B-Q4_K_M.gguf")
        
        if "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" not in llm_names:
            llm_names.append("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")

        #GEMMA
        if "gemma-3-270m-it-Q4_K_M.gguf" not in llm_names:
            llm_names.append("gemma-3-270m-it-Q4_K_M.gguf")

        if "gemma-3-1b-it-Q4_K_M.gguf" not in llm_names:
            llm_names.append("gemma-3-4b-it-Q4_K_M.gguf")

        if "gemma-3-4b-it-Q4_K_M.gguf" not in llm_names:
            llm_names.append("gemma-3-4b-it-Q4_K_M.gguf")

        #LLAMA
        if "Llama-3.2-1B-Instruct-Q4_K_M.gguf" not in llm_names:
            llm_names.append("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        
        if "Llama-3.2-3B-Instruct-Q4_K_M.gguf" not in llm_names:
            llm_names.append("Llama-3.2-3B-Instruct-Q4_K_M.gguf")

        #PHI
        if "Phi-4-mini-instruct-Q4_K_M.gguf" not in llm_names:
            llm_names.append("Phi-4-mini-instruct-Q4_K_M.gguf")
            
        
        
        return {'required': {'llm': (llm_names,), 'prompt': ('STRING', {'default': 'Woman in Paris', 'multiline': True})}}
        

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "text"
    OUTPUT_NODE = True

    def generate(self, llm, prompt):

        llm_path = os.path.join(llm_dir,llm)
        
        if os.path.isfile(llm_path) == False:

            match llm:

                #QWEN MODELS
                case "Qwen3-0.6B-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"Qwen3-0.6B-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
                
                case "Qwen3-1.7B-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"Qwen3-1.7B-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"

                case "Qwen3-4B-Instruct-2507-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
                
                # GEMMA MODELS
                case "gemma-3-270m-it-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"gemma-3-270m-it-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf"

                case "gemma-3-1b-it-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"gemma-3-1b-it-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
                
                case "gemma-3-4b-it-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"gemma-3-4b-it-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf"
                
                # LLAMA MODELS

                case "Llama-3.2-1B-Instruct-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"Llama-3.2-1B-Instruct-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

                case "Llama-3.2-3B-Instruct-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"Llama-3.2-3B-Instruct-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

                #PHI MODEL
                case "Phi-4-mini-instruct-Q4_K_M.gguf":
                    filename = os.path.join(llm_dir,"Phi-4-mini-instruct-Q4_K_M.gguf")
                    url = "https://huggingface.co/lmstudio-community/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf"

                case _:
                    return ("Unknown model", )

            try:
                # Send a GET request to the URL
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                
                # Open the local file to write the downloaded content
                with open(filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                
            except requests.exceptions.RequestException as e:
                return (f"Error downloading file: {e}", )
            

        try:
            
            chatformat = "chatml"
            if "gemma" in llm.lower():
                chatformat = "gemma"
            if "qwen" in llm.lower():
                chatformat = "qwen"
            if "llama" in llm.lower():
                chatformat = "llama-2"
            if "phi" in llm.lower():
                chatformat = "openchat"
            if "mistral" in llm.lower():
                chatformat = "mistral-instruct"
            
            print(chatformat)

            llm = Llama(
                model_path=llm_path,
                chat_format=chatformat,
                n_ctx=4096,
            )
            
            output = llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Try your best to give the best response possible to the user."},
                    {
                        "role": "user",
                        "content": """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like 'create an image').Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Always frame the scene, including details about the film grain, color grading, and any artifacts or characteristics specific. Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than 250 characters.
                        Description : """ + prompt
                    }
                ]
            )

            result = output["choices"][0]["message"]["content"]
            if "</think>" in result:
                result = result.split("</think>")[1]
            result = result.strip()

        except Exception as e:
            result = f"Error: Unable to generate caption. {str(e)}"

        print(result)
        return (result,)
    


NODE_CLASS_MAPPINGS = {"PromptBeautify": PromptBeautify}