"""Real torch/DataProto CPU checks; opt in on an installed training runtime."""
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('V83_NATIVE_CPU')=='1','requires installed native runtime')
class RuntimeNativeCpuTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU tests must hide CUDA devices')
        self.directory=tempfile.TemporaryDirectory()

    def tearDown(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        self.directory.cleanup()

    def config(self):
        return dict(env=dict(max_steps=50,rollout=dict(n=8)),algorithm=dict(gigpo=dict(enable_similarity=False)),
            repro_v83=dict(run_id='cpu-test',config_hash='frozen-test-config',model_contract_hash='frozen-model',
                output_dir=self.directory.name,enable_tier1a=True,enable_tier1b=True,enable_tier2=False))

    def test_native_dataproto_copy_reorder_mask_and_actual_token_std(self):
        import numpy as np
        import torch
        from verl import DataProto
        from repro_v83.runtime import RuntimeV83
        from repro_v83.credit import weighted_std
        runtime=RuntimeV83(self.config());runtime.start_iteration(1)
        runtime.action_groups={'a':'af','b':'af','m':'mixed'}
        runtime.credits={'centered':{'a':-1.,'b':1.,'m':0.},'af':{'af':True,'mixed':False}}
        runtime.phase='collected'
        # Duplicate a copied original row and reorder it as native adjustment
        # does; std must count actual token occurrences, not just unique UIDs.
        ids=np.array(['b','a','m','a'],dtype=object)
        mask=torch.tensor([[1.,0,0],[1.,1.,0],[1.,1.,1.],[1.,1.,0]])
        base=torch.arange(12,dtype=torch.float32).reshape(4,3)
        batch=DataProto.from_dict(tensors={'advantages':base.clone(),'response_mask':mask},
                                  non_tensors={'repro_action_uid':ids})
        original_mask=mask.clone();runtime.inject(batch)
        sigma=weighted_std([1.,-1.,-1.],[1.,2.,2.])
        scale=.5/(sigma+1e-6)
        expected=torch.tensor([1.,-1.,0.,-1.])[:,None]*scale*mask
        torch.testing.assert_close(batch.batch['advantages'],base+expected)
        self.assertTrue(torch.equal(batch.batch['response_mask'],original_mask))
        self.assertTrue(torch.equal(batch.batch['advantages'][2],base[2]))
        self.assertTrue(torch.equal(batch.batch['advantages'][mask==0],base[mask==0]))
        self.assertAlmostEqual(runtime.injection['af_token_std'],sigma)
        with self.assertRaises(RuntimeError):runtime.inject(batch)

    def test_checkpoint_roundtrip_and_contract_failure(self):
        from repro_v83.runtime import RuntimeV83
        runtime=RuntimeV83(self.config());runtime.start_iteration(5)
        runtime.phase='committed';runtime.optimizer_steps=17
        checkpoint=Path(self.directory.name)/'global_step_5';checkpoint.mkdir()
        cursors={'train':[{'seed':0,'reset_count':5}], 'validation':[{'seed':0,'reset_count':8}]}
        runtime.save(checkpoint,task_cursors=cursors)
        fresh=RuntimeV83(self.config());fresh.load(checkpoint)
        self.assertEqual((fresh.iteration_id,fresh.optimizer_steps),(5,17))
        self.assertEqual(fresh.task_cursors,cursors)
        config=self.config();config['repro_v83']['config_hash']='different'
        with self.assertRaises(ValueError):RuntimeV83(config).load(checkpoint)

    def test_pending_update_cannot_be_committed_or_checkpointed(self):
        from repro_v83.runtime import RuntimeV83
        runtime=RuntimeV83(self.config());runtime.start_iteration(1)
        with self.assertRaises(RuntimeError):runtime.start_iteration(2)
        with self.assertRaises(RuntimeError):runtime.save(Path(self.directory.name))


if __name__=='__main__':unittest.main()
