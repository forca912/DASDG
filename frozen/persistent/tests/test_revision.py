"""Synthetic software validation only. No outputs here are manuscript results."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model
from protocol import ROOT, load_data, mask_path, read_json, verify_complete, write_json, read_csv, lock
from train import evaluate_stream
from run import audit
from vendor.train_fdr_stgnn import FDRSTGNN
from vendor.train_v16b_external import GraphWaveNetAdaptive, ForecastWindowDataset, aggregate_metrics
from torch.utils.data import DataLoader


def tiny_config():
    c = read_json(ROOT / 'configs/primary.json')
    c['datasets'] = ['PEMS04']; c['seeds'] = [7]
    c['training'].update(hidden_dim=8, embedding_dim=4, graph_dim=4, top_k=2,
                         batch_size=8, epochs=2, patience=25, cpu_threads=1,
                         latency_warmup=1, latency_repetitions=2)
    return c


def fixture(root, c):
    rng = np.random.default_rng(123)
    flow = rng.uniform(1, 80, size=(144, 6)).astype(np.float32)
    split = [64, 104, 144]
    dest = root / 'processed/PEMS04/complete.npz'; dest.parent.mkdir(parents=True)
    np.savez_compressed(dest, flow=flow, split=split, mean=[flow[:64].mean()], std=[flow[:64].std()])
    masks = {s: rng.random((n, 6)) > .2 for s, n in [('train', 64), ('val', 40), ('test', 40)]}
    path = mask_path(root, 'PEMS04', 7, c); path.parent.mkdir(parents=True)
    np.savez_compressed(path, **masks, seed=7, block_length=12, intended_global_missing_rate=.1,
                        affected_node_fraction=.4, observed_is_one=True)
    return flow, masks


class Models(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(8)
        self.c = tiny_config()['training']; self.n=6
        self.a=np.eye(self.n, dtype=np.float32)
        self.x=torch.randn(2,12,self.n,3)
        self.x[...,1]=(torch.rand(2,12,self.n)>.2).float(); self.x[...,2]=self.x[...,2].abs()

    def model(self,name):return build_model(name,self.n,self.a,self.c)

    def test_published_prediction_and_gradients_preserved(self):
        legacy=FDRSTGNN(self.n,8,4,4,12,1,2,.1,.1,1.,'graph_no_reliability_channel',np.zeros(self.n),np.ones(self.n))
        revised=self.model('dasdg'); revised.load_state_dict(legacy.state_dict())
        legacy.train(); revised.train()
        torch.manual_seed(13); a=legacy(self.x)[0]; a.sum().backward()
        torch.manual_seed(13); b=revised(self.x); b.sum().backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for (n,p),(m,q) in zip(legacy.named_parameters(),revised.named_parameters()):
            self.assertEqual(n,m)
            if p.grad is None:self.assertIsNone(q.grad)
            else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_gwn_two_channel_preserved_and_three_channel_used(self):
        old=GraphWaveNetAdaptive(self.n,self.a,8,4,12,.1).eval()
        new=self.model('gwn_mask').eval(); new.load_state_dict(old.state_dict())
        torch.testing.assert_close(old(self.x),new(self.x),rtol=0,atol=0)
        three=self.model('gwn_duration').eval(); x=self.x.clone().requires_grad_()
        three(x).sum().backward()
        self.assertGreater(x.grad[...,2].abs().sum().item(),0)
        x=self.x.clone().requires_grad_();new(x).sum().backward()
        self.assertEqual(x.grad[...,2].abs().sum().item(),0)

    def test_single_factor_ablations(self):
        zero=self.model('dasdg_no_duration').eval(); altered=self.x.clone();altered[...,2]+=7
        torch.testing.assert_close(zero(self.x),zero(altered),rtol=0,atol=0)
        for name in tiny_config()['stages']['ablation']:
            model=self.model(name).eval(); model(self.x).sum().backward()
            if name=='dasdg_no_graph':
                self.assertIsNone(model.graph_value.weight.grad); self.assertIsNotNone(model.fusion_norm.weight.grad)
            if name=='dasdg_static':self.assertIsNone(model.dynamic_query.weight.grad)
            if name=='dasdg_dynamic':self.assertIsNone(model.static_query.weight.grad)
            if name=='dasdg_no_gate':self.assertIsNone(model.fusion_gate.weight.grad)
            if name=='dasdg_dense':self.assertTrue((model.last_adjacency>0).all())
        model=self.model('dasdg_static').eval(); model(self.x); a=model.last_adjacency.clone()
        model(self.x*2);torch.testing.assert_close(a,model.last_adjacency)
        self.assertTrue(((a>0).sum(-1)==2).all())

    def test_causal_inputs_and_streaming_metrics(self):
        rng=np.random.default_rng(5); flow=rng.uniform(1,40,(40,6)).astype(np.float32)
        mask=rng.random((40,6))>.3
        d=ForecastWindowDataset(flow,mask,20,5,12,12,4)
        changed=flow.copy();changed[12:]+=10000
        other=ForecastWindowDataset(changed,mask,20,5,12,12,4)
        torch.testing.assert_close(d[0][0],other[0][0])
        model=self.model('dasdg').eval(); loader=DataLoader(d,batch_size=4)
        rows=evaluate_stream(model,loader,torch.device('cpu'),5,20)
        with torch.no_grad():
            pred=np.concatenate([model(x).double().numpy()*5+20 for x,y in loader])
        target=np.concatenate([y.double().numpy()*5+20 for x,y in loader])
        expected=aggregate_metrics(pred,target)
        for k,v in expected.items():self.assertAlmostEqual(rows[-1][k],v,places=9)


class Integration(unittest.TestCase):
    def test_queue_resume_audit_and_corruption_rejection(self):
        qa=ROOT.parent.parent / 'qa'; qa.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='SMOKE_',dir=qa) as tmp:
            base=Path(tmp);c=tiny_config();cp=base/'config.json';write_json(cp,c)
            data=base/'data';fixture(data,c)
            common=['--config',str(cp),'--data-root',str(data),'--device','cpu','--fixture']
            queue=[sys.executable,'-u',str(ROOT/'run.py'),'run',*common,'--stage','all','--output',str(base/'SMOKE_all')]
            result=subprocess.run(queue,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=audit(base/'SMOKE_all',c,'all',fixture=True);self.assertEqual(report['completed'],10)
            original=base/'SMOKE_all/runs/PEMS04/seed7/dasdg'
            resumed=base/'SMOKE_resumed'
            cmd=[sys.executable,'-u',str(ROOT/'train.py'),*common,'--dataset','PEMS04','--seed','7','--model','dasdg','--output',str(resumed)]
            for extra in (['--stop-after-epoch','1'],[]):
                r=subprocess.run(cmd+extra,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            a=torch.load(original/'best.pt',weights_only=True);b=torch.load(resumed/'best.pt',weights_only=True)
            for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
            self.assertEqual(read_csv(original/'test_metrics.csv'),read_csv(resumed/'test_metrics.csv'))
            hist_a=read_csv(original/'history.csv');hist_b=read_csv(resumed/'history.csv')
            for a,b in zip(hist_a,hist_b):
                for k in ('epoch','train_loss_norm','val_mae_raw','learning_rate'):self.assertEqual(a[k],b[k])
            r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertIn('SKIP_VERIFIED_COMPLETE',r.stdout)
            with lock(base/'.test.lock'):
                with self.assertRaises(RuntimeError):
                    with lock(base/'.test.lock'):pass
            with (resumed/'test_metrics.csv').open('a') as f:f.write('\ncorrupt')
            with self.assertRaises(RuntimeError):verify_complete(resumed)
            modified=copy.deepcopy(c);modified['training']['epochs']=3;write_json(cp,modified)
            r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
            self.assertNotEqual(r.returncode,0)
            # Full PEMS preflight must reject tiny fixtures without --fixture.
            with self.assertRaises(ValueError):load_data(data,'PEMS04',7,c,fixture=False)


if __name__=='__main__':unittest.main(verbosity=2)
