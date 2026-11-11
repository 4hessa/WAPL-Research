"""CPU diagnostic of paired finite-horizon utility; not an LM benchmark."""
import json, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


def loss(w, x, y):
    return np.mean(np.logaddexp(0, x @ w) - y * (x @ w))


def grad(w, x, y):
    p = 1 / (1 + np.exp(-np.clip(x @ w, -50, 50)))
    return x.T @ (p - y) / len(y)


def update(w, velocity, x, y, lr=0.05):
    v = 0.9 * velocity + grad(w, x, y)
    return w - lr * v, v


def utility(w, v, candidate, follow, target):
    """Include one candidate versus skip; then identical continuation batches.

    Horizon counts the candidate slot, so h=1 has no continuation.
    Skip leaves optimizer state unchanged. This is insertion utility, not
    replacement utility or an equal-compute comparison.
    """
    wi, vi = update(w.copy(), v.copy(), *candidate)
    ws, vs = w.copy(), v.copy()
    for batch in follow:
        wi, vi = update(wi, vi, *batch)
        ws, vs = update(ws, vs, *batch)
    return float(loss(ws, *target) - loss(wi, *target))


def data(rng, n, teacher):
    x = rng.normal(size=(n, len(teacher)))
    y = (x @ teacher + rng.normal(size=n) * 0.3 > 0).astype(float)
    return x, y


def run(seed):
    rng = np.random.default_rng(seed)
    teacher = np.array([1., -1., .5, .2, -.4, .8])
    # Disjoint draws: candidate, probe, utility target and continuation.
    probe, target = data(rng, 96, teacher), data(rng, 128, teacher)
    w, v = rng.normal(size=6) * .2, np.zeros(6)
    rows = []
    for checkpoint in range(3):
        for _ in range(8):
            w, v = update(w, v, *data(rng, 8, teacher))
        follow = [data(rng, 8, teacher) for _ in range(3)]
        pg = grad(w, *probe)
        for _ in range(24):
            c = data(rng, 1, teacher)
            g = grad(w, *c)
            delta = -.05 * (.9 * v + g)
            baseline = [loss(w, *c), np.linalg.norm(g), g @ pg,
                        -delta @ pg, np.linalg.norm(delta)]
            extra = [w @ g, np.linalg.norm(delta)/(np.linalg.norm(w)+1e-12),
                     np.linalg.norm(w*g)]
            rows.append((checkpoint, baseline, extra, utility(w,v,c,follow,target)))
    return rows


def main():
    started = time.perf_counter()
    train, dev, test = list(range(10)), list(range(10,14)), list(range(100,110))
    records = {s:run(s) for s in train+dev+test}
    # Random control drawn independently of all labels, equal extra dimension.
    rng = np.random.default_rng(991)
    noise = {s:rng.normal(size=(72,3)) for s in records}
    def matrix(seeds, arm):
        xs, ys = [], []
        for s in seeds:
            for i, (_, b, e, y) in enumerate(records[s]):
                xs.append(b + (e if arm=='weight' else list(noise[s][i]) if arm=='random' else []));ys.append(y)
        return np.array(xs), np.array(ys)
    output = {}
    for arm in ['baseline','weight','random']:
        x,y = matrix(train,arm);xd,yd=matrix(dev,arm)
        scaler=StandardScaler().fit(x)
        choices=[]
        for alpha in [.1,1.,10.,100.]:
            model=Ridge(alpha=alpha).fit(scaler.transform(x),y)
            choices.append((np.mean(abs(model.predict(scaler.transform(xd))-yd)),alpha))
        alpha=min(choices)[1]
        model=Ridge(alpha=alpha).fit(scaler.transform(x),y)
        scores=[]
        for s in test:
            xt,yt=matrix([s],arm); pred=model.predict(scaler.transform(xt))
            # Rank within checkpoints, not pooled across changing model states.
            scores.append(float(np.mean([spearmanr(yt[j:j+24],pred[j:j+24]).statistic for j in range(0,72,24)])))
        xt,yt=matrix(test,arm)
        output[arm]={'alpha':alpha,'seed_spearman':scores,'mean_spearman':float(np.mean(scores)), 'mae':float(np.mean(abs(model.predict(scaler.transform(xt))-yt)))}
    diff=np.array(output['weight']['seed_spearman'])-np.array(output['baseline']['seed_spearman'])
    bootstrap=np.random.default_rng(2026).choice(diff,(5000,len(diff)),replace=True).mean(axis=1)
    output['comparison']={'weight_minus_baseline':float(diff.mean()),'seed_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist()}
    output['protocol']={'model':'6-parameter logistic regression; synthetic diagnostic only','horizon':4,'train_seeds':train,'dev_seeds':dev,'test_seeds':test,'elapsed_seconds':time.perf_counter()-started}
    Path('results').mkdir(exist_ok=True)
    Path('results/pilot.json').write_text(json.dumps(output,indent=2))
    Path('results/pilot-records.json').write_text(json.dumps(records))
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
