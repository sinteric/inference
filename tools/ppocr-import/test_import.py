import importlib.util
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import helper,numpy_helper

spec=importlib.util.spec_from_file_location('ppocr_import',Path(__file__).with_name('import.py'))
importer=importlib.util.module_from_spec(spec);spec.loader.exec_module(importer)


class ImportTest(unittest.TestCase):
    def test_constant_identity_reshape_and_shape(self):
        a=numpy_helper.from_array(np.arange(6,dtype=np.float32).reshape(2,3))
        shape=numpy_helper.from_array(np.array([3,2],np.int64))
        nodes=[helper.make_node('Constant',[],['a'],value=a),helper.make_node('Identity',['a'],['b']),
               helper.make_node('Constant',[],['shape'],value=shape),helper.make_node('Reshape',['b','shape'],['c']),
               helper.make_node('Shape',['c'],['d'])]
        m=helper.make_model(helper.make_graph(nodes,'test',[],[helper.make_tensor_value_info('d',onnx.TensorProto.INT64,[2])]))
        nodes,constants,_,_=importer.fold(m)
        self.assertEqual(nodes,[]);np.testing.assert_array_equal(constants['d'],[3,2])

    def test_unmatched_runtime_and_constant_ops_fail(self):
        node={'op':'Sin','inputs':['x'],'output':'y','attrs':{},'source_name':'unexpected'}
        with self.assertRaisesRegex(ValueError,'unmatched node'): importer.match_patterns([node],{},['y'])
        ns=[helper.make_node('Constant',[],['a'],value=numpy_helper.from_array(np.array([1],np.float32))),helper.make_node('Sin',['a'],['b'])]
        m=helper.make_model(helper.make_graph(ns,'test',[],[]))
        with self.assertRaisesRegex(ValueError,'unmatched constant'): importer.fold(m)

    def test_patterns_require_closed_expression(self):
        gate={'op':'HardSigmoid','inputs':['x'],'output':'gate','attrs':{'alpha':1/6,'beta':0.5},'source_name':'gate'}
        mul={'op':'Mul','inputs':['x','gate'],'output':'y','attrs':{},'source_name':'mul'}
        nodes=importer.match_patterns([gate,mul],{},['y'])
        self.assertEqual([n['op'] for n in nodes],['HardSwish'])
        # An exported gate remains live, so it must not be removed by fusion.
        mul=dict(mul,op='Mul',inputs=['x','gate'],attrs={})
        nodes=importer.match_patterns([gate,mul],{},['y','gate'])
        self.assertEqual([n['op'] for n in nodes],['HardSigmoid','Mul'])

    def test_resize_semantics_fail_closed(self):
        node={'op':'Resize','inputs':['x'],'output':'y','attrs':{'mode':b'linear'},'source_name':'resize'}
        with self.assertRaisesRegex(ValueError,'Resize semantics'): importer.match_patterns([node],{},['y'])

    def test_uvdoc_semantics_fail_closed(self):
        def node(op,attrs): return {'op':op,'inputs':['x'],'output':'y','attrs':attrs,'source_name':op.lower()}
        ok=[node('Resize',{'mode':b'linear','coordinate_transformation_mode':b'align_corners'}),
            node('Pad',{'mode':b'reflect'}),
            node('GridSample',{'mode':b'bilinear','padding_mode':b'zeros','align_corners':1})]
        for n in ok: self.assertEqual(importer.match_patterns([n],{},['y'],'uvdoc'),[n])
        bad=[node('Resize',{'mode':b'linear','coordinate_transformation_mode':b'half_pixel'}),
             node('Pad',{'mode':b'constant'}),
             node('GridSample',{'mode':b'bilinear','padding_mode':b'border','align_corners':1})]
        for n in bad:
            with self.assertRaisesRegex(ValueError,'semantics'): importer.match_patterns([n],{},['y'],'uvdoc')
        # The unwarper's operators stay out of the detector and recognizer.
        with self.assertRaisesRegex(ValueError,'unmatched node'): importer.match_patterns([ok[2]],{},['y'],'det')


if __name__=='__main__': unittest.main()
