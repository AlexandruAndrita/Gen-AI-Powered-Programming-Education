import openai
import project_part1_prompts as prompts
import project_part1_utils as utils

import os
import json

import torch

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

# Class for handling program repair using language models
class Repair:
    # Initialize the Repair class with model details
    def __init__(self, model_name, is_huggingface):
        self.model_name = model_name
        self.system_prompt = prompts.system_message_nus
        self.user_prompt = prompts.user_message_nus_repair_basic
        self.is_huggingface = is_huggingface
        
        if self.is_huggingface:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name, 
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="chatml",
            )
            FastLanguageModel.for_inference(self.model)
        else:
            with open("129_Andrita_openai.txt") as f:
                content = f.read().strip()
            openai_key = content.split("\n")[2]
            openai.api_key = openai_key

    # Extract fixed code from the generated text
    def extract_fixed_code(self, text):
        start_tag = "[FIXED]"
        end_tag = "[/FIXED]"
        
        start_index = text.find(start_tag)
        if start_index == -1:
            return ""
        
        start_index += len(start_tag)
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            end_index = len(text)
        
        extracted_text = text[start_index:end_index].strip()
        
        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.startswith("python"):
            extracted_text = extracted_text[6:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]
        
        return extracted_text

    # Save the transcript to a JSON file at "project_part1_transcripts/transcript.json". This file contains all prompts and LLM responses which can be used for debugging.
    def save_transcript_to_json(self, transcript):
        os.makedirs("project_part1_transcripts", exist_ok=True)
        file_path = os.path.join("project_part1_transcripts", "transcript.json")
        
        # Read existing data
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
        
        # Append new transcript data
        existing_data.extend(transcript)
        
        # Write back to the file
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

    # Call the OpenAI language model
    def call_llm_openai(self, system_prompt_formatted, user_prompt_formatted):
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt_formatted},
                {"role": "user", "content": user_prompt_formatted}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    # Call the Hugging Face language model
    def call_llm_huggingface(self, system_prompt_formatted, user_prompt_formatted):
        prompt_string = f"""<|system|>\n{system_prompt_formatted}<|end|>\n<|user|>\n{user_prompt_formatted}<|end|>\n<|assistant|>"""        
        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            max_new_tokens=2048, 
            use_cache=True,
            # You can use the parameters below to set the temperature for sampling. To learn more about these parameters, you can refer to https://huggingface.co/blog/how-to-generate.
            do_sample=True,
            temperature=0.7
        )
        
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        return response_text

    # Call the appropriate language model based on configuration
    def call_llm(self, problem_data, buggy_program):
        system_prompt_formatted = self.system_prompt
        user_prompt_formatted = self.user_prompt.format(problem_data=problem_data, buggy_program=buggy_program)
        
        transcript = []
        
        if self.is_huggingface:
            generated_response = self.call_llm_huggingface(system_prompt_formatted, user_prompt_formatted)
        else:
            generated_response = self.call_llm_openai(system_prompt_formatted, user_prompt_formatted)
        
        transcript.append({
            "input_prompt": system_prompt_formatted + user_prompt_formatted,
            "output": generated_response
        })
            
        self.save_transcript_to_json(transcript)
        
        return generated_response

    # Generate a repair for the given problem and program
    def generate_repair(self, problem_data, buggy_program, testcases):
        distance = utils.Distance()
        program_testcases = utils.Compiler()
        min_distance_achieved = float("inf")
        max_passed_testcases = float("-inf")
        best_repair = ""
        for i in range(3):
            # easier to keep track of the iterations in Colab
            print(f"Starting iteration {i}/3")
            try:
                generated_response = self.call_llm(problem_data, buggy_program)
                fixed_code = self.extract_fixed_code(generated_response)
                count_tests_passed = 0

                print(f"type = {type(fixed_code)}")
                print(f"type = {type(testcases)}")

                if fixed_code is None or fixed_code == "":
                    print(f"Index {i}/3 -- No valid fixed_code found. fixed_code = {fixed_code}")
                    continue

                all_tests_passed, results_testcases = program_testcases.run_program_with_testcases(fixed_code,
                                                                                                   testcases)
                print(f"Index {i}/3 -- All tests passed = {all_tests_passed}")

                for element in results_testcases:
                    if element[0]:
                        count_tests_passed += 1

                print(f"Index {i}/3 -- Passed {count_tests_passed}/{len(results_testcases)}")

                if all_tests_passed:
                    calculated_distance = distance.get_edit_distance(fixed_code, buggy_program)
                    if calculated_distance < min_distance_achieved:
                        best_repair = fixed_code
                        min_distance_achieved = calculated_distance
                else:
                    # 1st attempt - using the max number of passed tests
                    if count_tests_passed >= max_passed_testcases:
                        calculated_distance = distance.get_edit_distance(fixed_code, buggy_program)
                        if calculated_distance < min_distance_achieved:
                            best_repair = fixed_code
                            min_distance_achieved = calculated_distance
                            max_passed_testcases = count_tests_passed
                # elif count_tests_passed != 0:
                #     # 2nd attempt - using the distance only
                #     calculated_distance = distance.get_edit_distance(fixed_code, buggy_program)
                #     if calculated_distance < min_distance_achieved:
                #         best_repair = fixed_code
                #         min_distance_achieved = calculated_distance


            except Exception as e:
                print(f"Exception at iteration {i}/3 -- Exception = {e}")
                continue

            # easier to keep track of the iterations in Colab
            print(f"Ending iteration {i}/3")

        if best_repair is None:
            print("No valid repair found")
            return ""
        return best_repair

