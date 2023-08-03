from rouge_score import rouge_scorer

def initial_output_selection(responses, document, query):
    """ For initial output generation, if multiple outputs were generated, score them.
    Determine if any of the outputs meet the threshold -- if so, return this and indicate that no iterative improvement is required. 
    If not, then use preference reward model to select the best output having satisfied the most out of the 3 threshold components.
    If multiple outputs met all of the threshold criterion, then use preference reward model to select the best output. 
    """

    scores_dict = {}
    for response in responses:
        rouge1_recall, rougel_doc, rougel_query = scoring_metrics(response, document, query)
        scores_dict[response] = (rouge1_recall, rougel_doc, rougel_query)

    return find_best_from_threshold(scores_dict)


def iterate_output_selection(responses, document, query, prev_best_scores):
    """ For iterative improvement: if there are multiple greedy improvements, we score each output.
    Compare them with the previous best response: if all "improvements" are actually worse in all criterion, then re-generate (up to a user-specified # of times) (or just stop iterating).
    If all "improvements" are better than the previous response: if multiple or none are above the threshold, use preference reward model to select best.
    If only some of the criterion are better than the previous response, call preference reward model, including the previous best response. 
    If only 1 is above the threshold, then return this and indicate that no further iterative improvement is required.
    """

    if len(responses) > 1:
        # multiple greedy improvements
        scores_dict = {}
        for response in responses:
            rouge1_recall, rougel_doc, rougel_query = scoring_metrics(response, document, query)
            scores_dict[response] = (rouge1_recall, rougel_doc, rougel_query)
    else:
        scores_dict = {}
        rouge1_recall, rougel_doc, rougel_query = scoring_metrics(responses[0], document, query)
        scores_dict[responses[0]] = (rouge1_recall, rougel_doc, rougel_query)

    # if only 1 response, then best_response is just responses[0]
    # compare responses to threshold and select best
    best_response, best_response_scores, indicator, criterion_satisfied = find_best_from_threshold(scores_dict)
    if len(criterion_satisfied) == len(best_response_scores):
        # all criterion met, so threshold is reached
        return best_response, best_response_scores, True, criterion_satisfied, True

    # now compare to previous best response, if not above threshold
    prev_best_response = list(prev_best_scores.keys())[0]
    return compare_curr_to_prev(best_response, prev_best_response, best_response_scores, prev_best_scores, criterion_satisfied)


def compare_curr_to_prev(curr_best_response, prev_best_response, curr_best_response_scores, prev_best_scores, curr_criterion_satisfied):
    """ Compare the current best response to the previously saved best, and determine which is better -- if the new best has not reached the threshold.
    Return: best response, scores for the best response, an indicator of whether improvement is finished (always False here), list of sub-criterion met for the threshold, and an indicator of whether it improved
    """

    criterion_set = {'rouge1_recall', 'rougel_doc', 'rougel_query'}
    prev_best_rouge1_recall, prev_best_rougel_doc, prev_best_rougel_query = list(list(prev_best_scores.values())[0].keys())[0]
    curr_best_rouge1_recall, curr_best_rougel_doc, curr_best_rougel_query = curr_best_response_scores
    prev_criterion_satisfied = list(prev_best_scores.values())[0][(prev_best_rouge1_recall, prev_best_rougel_doc, prev_best_rougel_query)]

    if len(curr_criterion_satisfied) < len(prev_criterion_satisfied):
        # got worse / did not improve
        return prev_best_response, (prev_best_rouge1_recall, prev_best_rougel_doc, prev_best_rougel_query), False, prev_criterion_satisfied, False
    elif len(curr_criterion_satisfied) > len(prev_criterion_satisfied):
        # got better, so return the new one
        return curr_best_response, curr_best_response_scores, False, curr_criterion_satisfied, True
    else:
        # now need to examine the metrics to see if there was, in fact, an improvement
        if curr_criterion_satisfied == prev_criterion_satisfied:
            # if the same thresholds are still met, then let's check the remaining criterion -- select the one with majority of higher scores
            remaining_criterion = [criterion for criterion in criterion_set if criterion not in curr_criterion_satisfied]
            diff_list = []
            if 'rouge1_recall' in remaining_criterion:
                improve_diff_rouge1_recall = curr_best_rouge1_recall - prev_best_rouge1_recall
                diff_list.append(improve_diff_rouge1_recall)
            if 'rougel_doc' in remaining_criterion:
                improve_diff_rougel_doc = curr_best_rougel_doc - prev_best_rougel_doc
                diff_list.append(improve_diff_rougel_doc)
            if 'rougel_query' in remaining_criterion:
                improve_diff_rougel_query = curr_best_rougel_query - prev_best_rougel_query
                diff_list.append(improve_diff_rougel_query)
            improved_criterion_count = len([diff > 0 for diff in diff_list])
            if improved_criterion_count >= len(remaining_criterion) / 2:
                return curr_best_response, curr_best_response_scores, False, curr_criterion_satisfied, True
            else:
                return prev_best_response, (prev_best_rouge1_recall, prev_best_rougel_doc, prev_best_rougel_query), False, prev_criterion_satisfied, False
        else:
            # this means that the same # of criterion have been met, but they are not the same criterion as before
            # call preference RM?
            # for now, let's just return the new one
            return curr_best_response, curr_best_response_scores, False, curr_criterion_satisfied, True

    
def find_best_from_threshold(scores_dict):
    """ Using the results from comparison to threshold, determine which is the best output. This is using the number of threshold sub-criterion met,
    as well ranking on the basis of the preference reward model when it is unclear as to which output is preferable over the other as the new starting
    point for the subsequent round of iterative improvement.
    """

    (above_threshold, responses_above_threshold), (above_2, responses_above_2), (above_1, responses_above_1), (above_none, responses_above_none), criterion_satisfied = compare_to_threshold(scores_dict)
    if above_threshold == 1:
        # return True as an indicator that no improvement is required -- the current answer is sufficient
        return responses_above_threshold[0], scores_dict[responses_above_threshold[0]], True, criterion_satisfied[responses_above_threshold[0]]
    elif above_threshold > 1:
        # call reward model rank on responses_above_threshold and choose best, including True as the indicator since no improvement is necessary
        return 
    else:
        if above_2 == 1:
            return responses_above_2[0], scores_dict[responses_above_2[0]], False, criterion_satisfied[responses_above_2[0]]
        elif above_2 > 1:
            # call reward model rank on responses_above_2 and choose best, including False as the indicator since improvement is necessary
            return 
        else:
            if above_1 == 1:
                return responses_above_1[0], scores_dict[responses_above_1[0]], False, criterion_satisfied[responses_above_1[0]]
            elif above_1 >= 1:
                # call reward model rank on responses_above_1 and choose best, including False as the indicator since improvement is necessary
                return
            else:
                # no threshold components met -- call reward model rank on responses_above_none and choose best, including False is indicator since improvement is necessary
                if len(scores_dict) == 1:
                    return responses_above_none[0], scores_dict[responses_above_none[0]], False, criterion_satisfied[responses_above_none[0]]
                return 


def compare_to_threshold(scores_dict):
    """ Helper routine for comparison to the threshold on the 3 score components. Compute how many of the responses (either the # of initial outputs,
    or the # of principles, each with a greedy improvement) are above a particular number of sub-criterion.

    Return: tuples of counts to list of resposnses: (all_above_threshold, responses_above_threshold), (above_2, responses_above_2), (above_1, responses_above_1),
    (above_none, responses_above_none)
    """
    above_threshold = above_2 = above_1 = above_none = 0
    responses_above_threshold = responses_above_2 = responses_above_1 = responses_above_none = []
    criterion_satisfied = {}
    for response, scores in scores_dict.items():
        rouge1_recall, rougel_doc, rougel_query = scores

        rouge1_threshold = .02
        rougel_doc_threshold = .05
        rougel_query_threshold = .05

        threshold_count = 0
        criterion_list = []
        if rouge1_recall > rouge1_threshold:
            threshold_count += 1
            criterion_list.append('rouge1_recall')
        if rougel_doc > rougel_doc_threshold:
            threshold_count += 1
            criterion_list.append('rougel_doc')
        if rougel_query > rougel_query_threshold:
            threshold_count += 1
            criterion_list.append('rougel_query')
        criterion_satisfied[response] = criterion_list

        if threshold_count == 3:
            above_threshold += 1
            responses_above_threshold.append(response)
        elif threshold_count == 2:
            above_2 += 1
            responses_above_2.append(response)
        elif threshold_count == 1:
            above_1 += 1
            responses_above_1.append(response)
        else:
            above_none += 1
            responses_above_none.append(response)

    return (above_threshold, responses_above_threshold), (above_2, responses_above_2), (above_1, responses_above_1), (above_none, responses_above_none), criterion_satisfied


def scoring_metrics(generated_response, document, query):
    """ Scoring outputs for determining improvement and to be used as a stopping criterion for refinement. Scores include:
    * Rouge-1 recall between the generated response and the document, to measure specificity.
    * Rouge-L between the generated response and the document, to measure groundedness.
    * Rouge-L between the generated response and the query, to measure relevance / usefulness.
    """
    rouge1_response_doc = rouge1_score(generated_response, document)
    rouge1_f1, rouge1_precision, rouge1_recall = rouge1_response_doc

    rougel_response_doc = rougeL_score(generated_response, document)
    rougel_doc_f1, rougel_doc_precision, rougel_doc_recall = rougel_response_doc

    rougel_response_query = rougeL_score(generated_response, query)
    rougel_query_f1, rougel_query_precision, rougel_query_recall = rougel_response_query
    return rouge1_recall, rougel_doc_f1, rougel_query_f1


def rouge1_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure, scores["rouge1"].precision, scores["rouge1"].recall


def rougeL_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure, scores["rougeL"].precision, scores["rougeL"].recall


def reward_model_rank(reward_model, phase, outputs, context, document):
    # To be implemented
    return


"""
General structure:
* compare_to_threshold and reward_model_rank are standalone, along with rouge1_score and rougeL_score
* scoring_metrics calls rouge1_score and rougeL_score
* find_best_from_threshold calls compare_to_threshold and reward_model_rank
    * compare_to_threshold provides the counts and corresponding responses, and find_best_from_threshold uses that to choose the best output
    * reward_model_rank is called to break ties when the same number of stopping sub-criterion are met
* initial_output_selection and iterate_output_selection call scoring_metrics and find_best_from_threshold

"""
