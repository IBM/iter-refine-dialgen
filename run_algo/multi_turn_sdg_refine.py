import os
import argparse
import pandas as pd
import json
from tqdm import tqdm
from run_algo.utils import make_genai_request, make_hf_api_request
from run_algo.utils import count_string_tokens
from dotenv import load_dotenv
import statistics
import warnings
from rouge_score import rouge_scorer
from run_algo.prompt_setup_multi_turn import icl_setup
from run_algo.scoring import initial_output_selection, iterate_output_selection, scoring_metrics, find_best_from_threshold

def run(api_key, model, model_source, reward_model, task, num_samples, num_init_outputs, num_conv_turns, batch_size, max_attempts, dataset_path, init_icl_samples, output_path):
    # Iterative Improvement Algorithm for Response Refinement
    data = json.load(open(dataset_path))
    if model_source == 'ibm-generative-ai':
        if task == 'sdg':
            run_algo_multi_turn_genai(data, api_key, model, reward_model, num_samples, num_init_outputs, num_conv_turns, max_attempts, init_icl_samples, output_path)
        elif task == 'response':
            run_algo_genai(data, api_key, model, reward_model, num_samples, num_init_outputs, batch_size, max_attempts, init_icl_samples, output_path)   


def run_algo_multi_turn_genai(data, api_key, model, reward_model, num_samples, num_init_outputs, num_conv_turns, max_attempts, init_icl_samples, output_path):
    writer = open(output_path, 'a')
    principles = ['specific']
    total_iters_count = 0
    conv_turn_count = 0
    refinement = False

    for i, conv in enumerate(data[:]):
        if (i+1) > num_samples:
            break

        conv['context'] = conv['context'].split('\nAgent')[0]
        
        curr_bests = []
        curr_best_scores = []
        num_attempts = 0
        # need to update conv inside the while loop so we have the correct things saved
        while conv_turn_count < num_conv_turns or refinement:
            if not refinement:
                ## initial generation for a given query / conversation state (where the previous user query is not asking to refine)
                # either reach here at beginning of multi-turn process, or after generating a new user query

                init_gen_icl_prompt = icl_setup('initial generation', context=conv['context'], 
                                        document=conv['document'], principles=principles, prev_best_response=None, icl_examples=init_icl_samples)
                init_prompt_list = [init_gen_icl_prompt] * num_init_outputs

                outputs = make_genai_request(model, api_key, init_prompt_list, decoding_method="sample", max_new_tokens=800, min_new_tokens=1, stop_sequences=["|EoS|","\n\n"])
                
                curr_responses = [output['response'].strip() for output in outputs]
                curr_responses_fixed = []
                for response in curr_responses:
                    if '###' in response:
                        response = response.replace('###', '').strip()
                        curr_responses_fixed.append(response)
                    else:
                        curr_responses_fixed.append(response)

                ## scoring initial outputs
                curr_bests = []
                curr_best_scores = []
                if num_init_outputs > 1:
                    curr_best_response, curr_best_scores, indicator, criterion_satisfied = initial_output_selection(curr_responses_fixed, conv['document'], query=conv['context'])
                    curr_bests.append(curr_best_response)
                    curr_best_scores.append({curr_best_response: {curr_best_scores: criterion_satisfied}})
                else:
                    rouge1_recall, rougel_doc, rougel_query = scoring_metrics(curr_responses_fixed[0], conv['document'], query=conv['context'])
                    _, _, indicator, criterion_satisfied = find_best_from_threshold({curr_responses_fixed[0]: (rouge1_recall, rougel_doc, rougel_query)})
                    curr_best_response = curr_responses_fixed[0]
                    curr_bests.append(curr_best_response)
                    curr_best_scores.append({curr_responses_fixed[0]: {(rouge1_recall, rougel_doc, rougel_query): criterion_satisfied}})

                # now determine if refinement is necessary or not
                if indicator:
                    # the current answer is acceptable (above the threshold), so we move onto generating a user query

                    curr_context = conv['context']
                    updated_context = curr_context + f'\nAgent: {curr_best_response}'
                    conv['context'] = updated_context
                    conv_turn_count += 1
                    if conv_turn_count >= num_conv_turns:
                        # cannot go further in utterances
                        break

                    # generate new user query
                    user_query_prompt = icl_setup('user query', context=conv['context'], document=conv['document'], principles=principles, 
                                            prev_best_response=None, icl_examples=None)
                                            
                    user_query_output = make_genai_request(model, api_key, [user_query_prompt], decoding_method="sample", 
                                   max_new_tokens=800, min_new_tokens=1, stop_sequences=["|EoS|","\n\n"])
                    user_query = user_query_output[0]['response'].strip()

                    # append current answer (curr_best_response) and the new user query to the end of conv
                    curr_context = conv['context']
                    updated_context = curr_context + f'\nUser: {user_query}'
                    conv['context'] = updated_context
                    conv_turn_count += 1
                    if conv_turn_count >= num_conv_turns:
                        # cannot go further in utterances
                        break
                else:
                    # we need to refine our answer
                    # update the conversation history (conv) to include curr_best_response and "Make more specific" user query
                    refinement = True
                    num_attempts = 0
                    prev_context = conv['context']
                    curr_context = conv['context']
                    updated_context = curr_context + f'\nAgent: {curr_best_response}\nUser: Please make your response more specific.'
                    conv['context'] = updated_context
                    conv_turn_count += 2

            else:
                # do refinement now
                indicator = False
                while (num_attempts < max_attempts and not indicator):
                    # get one prompt for each principle
                    iter_improve_icl_prompts = icl_setup('iterative improvement multi-turn', context=prev_context, document=conv['document'], 
                                             principles=principles, prev_best_response=curr_bests[-1])
                    
                    iterate_outputs = make_genai_request(model, api_key, iter_improve_icl_prompts, decoding_method="sample", max_new_tokens=800, min_new_tokens=1, stop_sequences=["|EoS|","\n\n"])
                    curr_iter_responses = [output['response'].strip() for output in iterate_outputs]
                    curr_iter_responses_fixed = []
                    for response in curr_iter_responses:
                        if '###' in response:
                            response = response.replace('###', '').strip()
                            curr_iter_responses_fixed.append(response)
                        else:
                            curr_iter_responses_fixed.append(response)
                    
                    # scoring refinements        
                    new_best_response, new_best_scores, indicator, criterion_satisfied, improved = iterate_output_selection(curr_iter_responses_fixed, conv['document'], conv['context'], curr_best_scores[-1])


                    if improved:
                        curr_bests.append(new_best_response)
                        curr_best_scores.append({new_best_response: {new_best_scores: criterion_satisfied}})

                        if not indicator and (num_attempts + 1 < max_attempts):
                            # should only update context if improved (otherwise, conversation history is the same)
                            # if we have to keep refining after this, append curr_bests[-1] and "Make more specific" user query to the end 
                            prev_context = conv['context']
                            curr_context = conv['context']
                            updated_context = curr_context + f'\nAgent: {curr_bests[-1]}\nUser: Please make your response more specific.'
                            conv['context'] = updated_context
                            conv_turn_count += 2
                    num_attempts += 1

                # update the best response to the end of conv
                curr_context = conv['context']
                updated_context = curr_context + f'\nAgent: {curr_bests[-1]}'
                conv['context'] = updated_context
                conv_turn_count += 1
                if conv_turn_count >= num_conv_turns:
                    # cannot go further in utterances
                    break

                # generate new user query
                user_query_prompt = icl_setup('user query', context=conv['context'], document=conv['document'], principles=principles, 
                                        prev_best_response=None, icl_examples=None)
                user_query_output = make_genai_request(model, api_key, [user_query_prompt], decoding_method="sample", 
                                max_new_tokens=800, min_new_tokens=1, stop_sequences=["|EoS|","\n\n"])
                user_query = user_query_output[0]['response'].strip()
                       
                # append new user query to conv                                                                                                     
                curr_context = conv['context']
                updated_context = curr_context + f'\nUser: {user_query}'
                conv['context'] = updated_context
                conv_turn_count += 1
                if conv_turn_count >= num_conv_turns:
                    # cannot go further in utterances
                    break

                refinement = False

        # multi-turn dialogue is finished so now let's see the full generated dialogue
        print(conv['context'])

        # reset to move onto the next dialogue / conversation history
        conv_turn_count = 0    
        refinement = False


def run_algo_genai(data, api_key, model, reward_model, num_samples, num_init_outputs, batch_size, max_attempts, init_icl_samples, output_path):
    writer = open(output_path, 'a')
    batched_inputs = []

    principles = ['specific']
    sample_count = 0
    total_iters_count = 0

    unanswerable = []
    unanswerable_count = 0

    for i, conv in enumerate(data[:]):
        if (i+1) > num_samples:
            break

        #if sample_count > num_samples:
        #    break

        if (i+1) % batch_size != 0:
            batched_inputs.append(conv)
            sample_count += 1 
            continue

        #if conv['type'] == 'negative':
            #continue
        
        batched_inputs.append(conv)
        #sample_count += 1

        # ### Determining answerability
        # icl_answerable_prompts = []
        # for conv in batched_inputs:
        #     answerability_prompt = icl_setup('answerable', context=conv['context'], document=conv['document'], principles=principles, 
        #                                      prev_best_response=None, icl_examples=None)
        #     icl_answerable_prompts.append(answerability_prompt)
        # answerability_outputs = make_genai_request(model, api_key, icl_answerable_prompts, decoding_method="sample",
        #                              max_new_tokens=100, min_new_tokens=1, stop_sequences=["|EoS|","\n\n"])
        # cleaned_answerability_outputs = []
        # for output in answerability_outputs:
        #     if '###' in output['response']:
        #         cleaned_answerability_outputs.append(output['response'].split('###')[0])
        #         print("Answerability outcome: ", output['response'].split('###')[0])
        #     else:
        #         cleaned_answerability_outputs.append(output['response'])
        #         print('Answerability outcome: ', output['response'])
        # for j, output in enumerate(cleaned_answerability_outputs):
        #     if 'negative' in output:
        #         unanswerable.append(batched_inputs[j])
        
        # if len(unanswerable) != 0:
        #     unanswerable_count += len(unanswerable)
        #     for convo in unanswerable:
        #         print(f"\nDocument: {convo['document']} \n\nContext: {convo['context']} \n\nlabel: negative (Not Answerable)")
        #     batched_inputs = []
        #     unanswerable = []
        #     continue

        #### Initial Output Generation ####
        batch_icl_prompts = []
        init_token_counts = {}
        final_responses = {}

        ## Storing a dictionary of the context and the document to the full conversation data
        batch_dict = {}
        for conv in batched_inputs:
            batch_dict[(conv['context'], conv['document'])] = conv

        ## Get prompts with in-context examples for initial generation, and generate outputs
        for batch_conv in batched_inputs:
            init_gen_icl_prompt = icl_setup('initial generation', context=batch_conv['context'], 
                                        document=batch_conv['document'], principles=principles, prev_best_response=None, icl_examples=init_icl_samples)
            init_prompt = [init_gen_icl_prompt] * num_init_outputs
            batch_icl_prompts.extend(init_prompt)
            init_token_counts[batch_conv['context']] = count_string_tokens(init_gen_icl_prompt)
        outputs = make_genai_request(model, api_key, batch_icl_prompts, decoding_method="sample", 
                                   max_new_tokens=800, min_new_tokens=1, stop_sequences=["|EoS|","\n\n"])
        
        # size of outputs is batched_inputs * num_init_outputs
        ## storing initial responses, adjustment for num_init_outputs (exploration in initial output generation)
        init_generated_responses = {}
        count = 0
        for j in range(len(batched_inputs)):
            curr_conv = batched_inputs[j]
            conv_outputs = outputs[count:count+num_init_outputs]
            curr_responses = [output['response'].strip() for output in conv_outputs]
            curr_responses_fixed = []
            for response in curr_responses:
                if '###' in response:
                    response = response.replace('###', '').strip()
                    curr_responses_fixed.append(response)
                else:
                    curr_responses_fixed.append(response)
            init_generated_responses[(curr_conv['context'], curr_conv['document'])] = curr_responses_fixed
            count += num_init_outputs

        #### Determine Best Initial Response ####
        curr_best_dict = {}
        best_init_generations = []
        best_scores_dict = {}
        remove_after_init_gen = []    # for the outputs that already meet threshold after initial generation
        if num_init_outputs > 1:
            for conv_info, curr_conv_outputs in init_generated_responses.items():
                context, document = conv_info

                ## Scoring -- partially implemented
                curr_best_response, curr_best_scores, indicator, criterion_satisfied = initial_output_selection(curr_conv_outputs, document, query=context)
                curr_best_dict[(context, document)] = [curr_best_response]
                best_scores_dict[(context, document)] = [{curr_best_response: {curr_best_scores: criterion_satisfied}}]
                best_init_generations.append(curr_best_response)
                # handle indicator for ending iterative improvement
                if indicator:
                    curr_full_conv = batch_dict[(context, document)]
                    conv_number = batched_inputs.index(curr_full_conv) + i + 1 - batch_size
                    remove_after_init_gen.append(curr_full_conv)
                    final_responses[conv_number] = curr_best_response

        else:
            for conv_info, outputs in init_generated_responses.items():
                context, document = conv_info

                # Scoring method goes here
                curr_best_dict[(context, document)] = [outputs[0]]
                rouge1_recall, rougel_doc, rougel_query = scoring_metrics(outputs[0], document, query=context)
                _, _, indicator, criterion_satisfied = find_best_from_threshold({outputs[0]: (rouge1_recall, rougel_doc, rougel_query)})
                best_scores_dict[(context, document)] = [{outputs[0]: {(rouge1_recall, rougel_doc, rougel_query): criterion_satisfied}}]
                best_init_generations.append(outputs[0])
                # handle indicator for ending iterative improvement
                if indicator:
                    curr_full_conv = batch_dict[(context, document)]
                    conv_number = batched_inputs.index(curr_full_conv) + i + 1 - batch_size
                    remove_after_init_gen.append(curr_full_conv)
                    final_responses[conv_number] = outputs[0]
        
        for conversation in batched_inputs:
            prompt_token_count = init_token_counts[conversation['context']]
            generated_token_count = count_string_tokens(curr_best_dict[(conversation['context'], conversation['document'])][-1])
            init_token_counts[conversation['context']] = prompt_token_count + generated_token_count

        # Remove conversations from batched_inputs if already reached threshold
        mod_batch = [conv for conv in batched_inputs if conv not in remove_after_init_gen]

        #### Iterative Refinement / Improvement Step ####
        num_attempts = 1
        while num_attempts <= max_attempts and len(mod_batch) != 0:
            batch_improve_prompts = []
            if num_attempts == 1:
                iter_token_counts = {}
            remove_in_iter = []  # remove from mod_batch if reached threshold

            for batched_conv in mod_batch:
                iter_improve_icl_prompts = icl_setup('iterative improvement', context=batched_conv['context'], document=batched_conv['document'], 
                                             principles=principles, prev_best_response=curr_best_dict[(batched_conv['context'], batched_conv['document'])][-1])
                #iter_improve_icl_prompts = icl_setup('iterative improvement zero shot', context=batched_conv['context'], document=batched_conv['document'],
                                                 #principles=principles, prev_best_response=curr_best_dict[(batched_conv['context'], batched_conv['document'])][-1])
                curr_tok_count = [statistics.fmean([count_string_tokens(prompt) for prompt in iter_improve_icl_prompts])]
                batch_improve_prompts.extend(iter_improve_icl_prompts)
                principle_count = len(iter_improve_icl_prompts)
                if num_attempts == 1:
                    iter_token_counts[batched_conv['context']] = curr_tok_count
                else:
                    prev_tok_counts = iter_token_counts[batched_conv['context']]
                    prev_tok_counts.extend(curr_tok_count)
                    iter_token_counts[batched_conv['context']] = prev_tok_counts
            iterate_outputs = make_genai_request(model, api_key, batch_improve_prompts, decoding_method="sample", max_new_tokens=800, min_new_tokens=1, stop_sequences=["|EoS|","\n\n"])
        
            iter_generated_responses = {}
            count = 0
            for j in range(len(mod_batch)):
                curr_conv = mod_batch[j]
                conv_iter_outputs = iterate_outputs[count:count+principle_count]
                curr_iter_responses = [output['response'].strip() for output in conv_iter_outputs]
                curr_iter_responses_fixed = []
                for response in curr_iter_responses:
                    if '###' in response:
                        response = response.replace('###', '').strip()
                        curr_iter_responses_fixed.append(response)
                    else:
                        curr_iter_responses_fixed.append(response)
                iter_generated_responses[(curr_conv['context'], curr_conv['document'])] = curr_iter_responses_fixed
                count += principle_count
        
            ## Scoring method for improvement
            for conv_info, iter_outputs in iter_generated_responses.items():
                context, document = conv_info
                new_best_response, new_best_scores, indicator, criterion_satisfied, improved = iterate_output_selection(iter_outputs, document, context,
                                                                                                                        best_scores_dict[(context, document)][-1])
                if improved:
                    best_responses_curr_conv = curr_best_dict[(context, document)]
                    best_responses_curr_conv.append(new_best_response)
                    curr_best_dict[(context, document)] = best_responses_curr_conv

                    best_scores_curr_conv = best_scores_dict[(context, document)]
                    best_scores_curr_conv.append({new_best_response: {new_best_scores: criterion_satisfied}})
                    best_scores_dict[(context, document)] = best_scores_curr_conv
                # handle indicator if no more improvement needed
                if indicator:
                    curr_full_conv = batch_dict[(context, document)]
                    conv_number = batched_inputs.index(curr_full_conv) + i + 1 - batch_size
                    remove_in_iter.append(curr_full_conv)
                    final_responses[conv_number] = new_best_response
                else:
                    if num_attempts == max_attempts:
                        curr_full_conv = batch_dict[(context, document)]
                        conv_number = batched_inputs.index(curr_full_conv) + i + 1 - batch_size
                        final_responses[conv_number] = new_best_response
     
            for conversation in mod_batch:
                prompt_token_count = iter_token_counts[conversation['context']][-1]
                generated_token_count = count_string_tokens(curr_best_dict[(conversation['context'], conversation['document'])][-1])
                curr_tok_history = iter_token_counts[conversation['context']]
                curr_tok_history.append(prompt_token_count + generated_token_count)
                iter_token_counts[conversation['context']] = curr_tok_history
            num_attempts += 1
            temp = [conv for conv in mod_batch if conv not in remove_in_iter]
            mod_batch = temp

            if num_attempts > max_attempts:
                for j, conv in enumerate(mod_batch):
                    conv_number = batched_inputs.index(conv) + i + 1 - batch_size
                    final_responses[conv_number] = curr_best_dict[(conv['context'], conv['document'])][-1]
        total_iters_count += (num_attempts-1)

        for j, convo in enumerate(batched_inputs):
            # For printing -- curr_best_dict has list of the best response for each iteration
            # best_scores_dict has list of the scores corresponding to the best response for each iteration
            print(f"\n\n**** Instance {i-batch_size+j+1} ****")
            print(f"\n{0} Document: {convo['document']} \n\nContext: {convo['context']} \n\nAgent: {best_init_generations[j]}  | Token Count: {init_token_counts[conv['context']]}")
            init_scores = list(best_scores_dict[(convo['context'], convo['document'])][0][best_init_generations[j]].keys())[0]
            print(f"\nScores -- Rouge-1 Recall (response-document): {init_scores[0]} | Rouge-L F1 (response-document): {init_scores[1]} | Rouge-L F1 (response-query): {init_scores[2]}")
            
            scores_list = []
            scores_list.append(init_scores)
            for k, response in enumerate(curr_best_dict[(convo['context'], convo['document'])]):
                if k != 0:
                    print(f"\n{k} Agent: {response}  |  Token Count: {iter_token_counts[conv['context']][k]}")
                    curr_scores = list(best_scores_dict[(conv['context'], convo['document'])][k][response].keys())[0]
                    scores_list.append(curr_scores)
                    print(f"\n Scores -- Rouge-1 Recall (response-document): {curr_scores[0]} | Rouge-L F1 (response-document): {curr_scores[1]} | Rouge-L F1 (response-query): {curr_scores[2]}")

            # write dict to the json file -- context and document, gold response, initial response (from best_init_generations) and final response (from final_responses)
            conv_output_dict = {"document": convo['document'], "context": convo['context'], "gold_response": convo['response'], "initial_response": best_init_generations[j], 
                                "final_response": final_responses[i-batch_size+j+1]}
            writer.write(json.dumps(conv_output_dict) + '\n')
            
        batched_inputs = []

    print("\n\n Average number of refinement iteration attempts: ", (total_iters_count / num_samples))
    #print("Unanswerable count: ",unanswerable_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='google/flan-ul2', help='Model to query')
    parser.add_argument("--task", type=str, default='sdg', help="task: either single-turn response generation ('response') or multi-turn dialogue ('sdg')")
    parser.add_argument("--reward_model", type=str, default='OpenAssistant/reward-model-deberta-v3-large-v2', help='Reward model for ranking')
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples in the dataset to run")
    parser.add_argument("--num_init_outputs", type=int, default=1, 
                        help='Number of initial outputs to obtain for choosing best starting response')
    parser.add_argument("--num_conv_turns", type=int, default=1, help='number of conversation turns to simulate')
    parser.add_argument("--batch_size", type=int, default=1, help='number of samples for a batch, set to 1')
    parser.add_argument("--max_attempts", type=int, default=2, help='maximum number of improvement attempts ')
    parser.add_argument("--dataset_path", type=str, default='data/md2d/subdocs_data/md2d_subdocs_val_pos_neg.json', help="Dataset to run algo on")
    parser.add_argument("--init_icl_samples", type=str, default='data/icl_samples/icl_init_gen.jsonl', 
                        help='ICL samples for initial response generation')
    parser.add_argument("--model_source", type=str, default='ibm-generative-ai', help='source of model')
    parser.add_argument("--output_path", type=str, default='.output', help='path for output')

    args = parser.parse_args()
    output_path = args.model.split('/')[1] + '-' + str(args.num_samples) + '-samples' + args.output_path
    load_dotenv()
    if args.model_source == 'ibm-generative-ai':
        api_key = os.getenv("GENAI_KEY", None)
    print(args.model)
    run(api_key, args.model, args.model_source, args.reward_model, args.task, args.num_samples, args.num_init_outputs, args.num_conv_turns, args.batch_size,
        args.max_attempts, args.dataset_path, args.init_icl_samples, output_path)
