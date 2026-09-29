import math, json, unittest
import numpy as np
import ouvidoria_core as ouvi
class TestOuvidoria(unittest.TestCase):
 def test_pares_sem_diagonal_e_limiar_estrito(self):
  m=np.array([[1,.9,.85],[.9,1,.2],[.85,.2,1]])
  self.assertEqual(ouvi.pares_da_matriz(m,.85,['a','b','c']),[{'id_a':'a','id_b':'b','similaridade':.9}])
 def test_metricas(self):
  resultado=ouvi.avaliar_pares([{'id_a':'a','id_b':'b'},{'id_a':'a','id_b':'c'}],[['a','b'],['b','c']],['a','b','c'])
  self.assertEqual(resultado['FP'],[('a','c')]); self.assertEqual(resultado['FN'],[('b','c')])
  self.assertEqual(resultado['precision'],.5); self.assertEqual(resultado['recall'],.5)
 def test_chunking_limites_e_offsets(self):
  texto=('A rua precisa de manutenção. A iluminação não funciona. '*25).strip()
  for config in [(240,0),(240,60),(400,80)]:
   cs=ouvi.dividir(texto,*config)
   self.assertGreater(len(cs),1)
   cobertos=set()
   for c in cs:
    self.assertLessEqual(c['caracteres'],config[0]); self.assertGreaterEqual(c['inicio'],0)
    self.assertEqual(texto[c['inicio']:c['inicio']+c['caracteres']],c['texto'])
    cobertos.update(range(c['inicio'],c['inicio']+c['caracteres']))
   self.assertTrue(all(i in cobertos for i,ch in enumerate(texto) if not ch.isspace()))
  with self.assertRaises(ValueError): ouvi.dividir(texto,100,100)
 def test_projecao_um_ponto(self):
  np.testing.assert_array_equal(ouvi.projetar([[1,2,3]],'PCA'),[[0,0]])
 def test_validacao_json(self):
  import io
  base=[{'id':'M001','data':'2026-01-01','categoria_oficial':'saúde','texto':'Texto de teste unitário'}]
  self.assertEqual(len(ouvi.carregar_base(io.StringIO(json.dumps(base)))),1)
  with self.assertRaises(ValueError): ouvi.carregar_base(io.StringIO(json.dumps(base*2)))
if __name__=='__main__': unittest.main(verbosity=2)
