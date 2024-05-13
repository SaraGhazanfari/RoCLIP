import time
from math import ceil

import numpy as np
import torch
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, batch_text,
                 min_generation_length, max_generation_length, num_beams, length_penalty):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.batch_text = batch_text
        self.min_generation_length = min_generation_length
        self.max_generation_length = max_generation_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        # self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        print('cHat is found!')
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        b_pABar = self._bootstrap(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            b_radius = self.sigma * norm.ppf(b_pABar)
            return cAHat, radius, b_radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        # self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        start_time = time.time()
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch) * self.sigma

                outputs, scores = self.base_classifier.get_outputs_and_scores(batch_images=batch + noise,
                                                                              batch_text=self.batch_text,
                                                                              min_generation_length=self.min_generation_length,
                                                                              max_generation_length=self.max_generation_length,
                                                                              num_beams=self.num_beams,
                                                                              length_penalty=self.length_penalty)
                scores = scores[0:1, :]
                counts += self._count_arr(scores.argmax(1).cpu().numpy(), self.num_classes)
                if _ == 0:
                    print(f'{(time.time() - start_time) * num / 60} minutes')
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    import numpy as np

    # Example data: counts of categories
    data = np.array([1000, 2000, 7000])  # e.g., 100 for category 1, 200 for category 2, 700 for category 3

    # Number of bootstrap samples
    num_samples = 1000

    # Array to store bootstrap probability estimates
    bootstrap_probs = np.zeros((num_samples, len(data)))

    # Perform bootstrapping
    for i in range(num_samples):
        bootstrap_sample = np.random.multinomial(n=sum(data), pvals=data / sum(data))
        bootstrap_probs[i] = bootstrap_sample / sum(bootstrap_sample)

    alpha = 0.001

    # Calculate mean and 95% confidence intervals for each category
    mean_probs = np.mean(bootstrap_probs, axis=0)
    ci_lower = np.percentile(bootstrap_probs, alpha / 2 * 100, axis=0)
    ci_upper = np.percentile(bootstrap_probs, 100 - alpha / 2 * 100, axis=0)

    # Print results
    for i, (mean, lower, upper) in enumerate(zip(mean_probs, ci_lower, ci_upper)):
        print(f"Category {i + 1}: Mean p_{i + 1} = {mean:.3f}, {100 - alpha * 100}% CI = [{lower:.3f}, {upper:.3f}]")

    def _bootstrap(self, nA, n, alpha):
        import numpy as np
        num_samples = 1000

        # Array to store bootstrap probability estimates
        bootstrap_probs = np.zeros((num_samples, 2))

        # Perform bootstrapping
        for i in range(num_samples):
            bootstrap_sample = np.random.multinomial(n=n, pvals=[nA / n, 1 - (nA / n)])
            bootstrap_probs[i] = bootstrap_sample / sum(bootstrap_sample)

        # Calculate mean and 95% confidence intervals for each category
        mean_probs = np.mean(bootstrap_probs, axis=0)
        ci_lower = np.percentile(bootstrap_probs, alpha / 2 * 100, axis=0)
        ci_upper = np.percentile(bootstrap_probs, 100 - alpha / 2 * 100, axis=0)

        # Print results
        for i, (mean, lower, upper) in enumerate(zip(mean_probs, ci_lower, ci_upper)):
            print(
                f"Category {i + 1}: Mean p_{i + 1} = {mean:.3f}, {100 - alpha * 100}% CI = [{lower:.3f}, {upper:.3f}]")

        return ci_lower[0]
