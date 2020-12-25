import tensorflow as tf
import os
import glob
import shutil


def keep_best_model(estimator, best_result, new_eval_result, best_checkpoint_path="best/"):
    if new_eval_result > best_result:
        tf.logging.info(
            'Saving a new better model ({:.3f} better than {:.3f})...'.format(new_eval_result, best_result))
        # copy the checkpoints files *.meta *.index, *.data* each time there is a better result, no cleanup for max
        # amount of files here
        latest_checkpoint = estimator.latest_checkpoint()

        # remove previous best dir
        for previous_best_dir in glob.iglob(os.path.join(os.path.dirname(latest_checkpoint), best_checkpoint_path, '*')):
            print(previous_best_dir)
            if os.path.exists(previous_best_dir):
                shutil.rmtree(previous_best_dir)

        for name in glob.glob(latest_checkpoint + '.*'):
            # copy current best content to best dir
            copy_to = os.path.join(os.path.dirname(latest_checkpoint), best_checkpoint_path + "{:.3f}".format(new_eval_result))
            if not os.path.exists(copy_to):
                os.makedirs(copy_to)
            shutil.copy(name, os.path.join(copy_to, os.path.basename(name)))
        # also save the text file used by the estimator api to find the best checkpoint
        with open(os.path.join(copy_to, "checkpoint"), 'w+') as f:
            f.write("model_checkpoint_path: \"{}\"".format(os.path.basename(latest_checkpoint)))
        best_result = new_eval_result
    return best_result