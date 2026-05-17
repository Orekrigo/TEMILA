import numpy as np
import torch
import torch.nn.functional as F


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(max(temperature, 1e-6))
    return np.asarray(logits, dtype=np.float64) / temperature


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    logits_t = torch.tensor(np.asarray(logits, dtype=np.float32))
    labels_t = torch.tensor(np.asarray(labels, dtype=np.int64))
    log_temperature = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=int(max_iter), line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).detach().cpu().item())
    return max(temperature, 1e-6)


def calibrated_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = temperature_scale_logits(logits, temperature)
    return softmax_np(scaled)


def confidence_features(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = np.asarray(prob, dtype=np.float64)
    order = np.argsort(-prob, axis=1)
    top1 = prob[np.arange(len(prob)), order[:, 0]]
    top2 = prob[np.arange(len(prob)), order[:, 1]] if prob.shape[1] > 1 else top1
    margin = top1 - top2
    entropy = -(prob * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1) / np.log(prob.shape[1])
    return top1, margin, entropy
