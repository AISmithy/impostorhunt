import os

def is_model_saved(model_dir):
    """
    Checks if the model and tokenizer are already saved in the given directory.
    Returns True if both model and tokenizer files exist, else False.
    """
    required_files = [
        os.path.join(model_dir, 'config.json'),
        # os.path.join(model_dir, 'pytorch_model.bin'),
        os.path.join(model_dir, 'tokenizer_config.json'),
        os.path.join(model_dir, 'vocab.txt'),
        os.path.join(model_dir, 'special_tokens_map.json'),
    ]
    return all(os.path.isfile(f) for f in required_files)


def summarize_test_results(test_results):
    """
    Summarizes the predictions from test results.
    Args:
        test_results (list of dict): Each dict should have 'article', 'prediction', and 'confidence'.
    Returns:
        dict: Summary with total, real_count, fake_count, and average confidence.
    """
    summary = {
        'total': len(test_results),
        'real_count': 0,
        'fake_count': 0,
        'average_confidence': 0.0
    }
    if not test_results:
        return summary
    real, fake, conf_sum = 0, 0, 0.0
    for res in test_results:
        pred = res['prediction']
        conf = res['confidence']
        if 'file_1.txt is REAL' in pred:
            real += 1
        elif 'file_2.txt is REAL' in pred:
            fake += 1
        conf_sum += conf
    summary['real_count'] = real
    summary['fake_count'] = fake
    summary['average_confidence'] = conf_sum / len(test_results)
    return summary
