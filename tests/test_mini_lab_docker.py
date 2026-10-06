import asyncio
import os
import unittest

from src.rvl_systems.lab.coding import CodingTask,DockerCodingGrader,IOTest
from src.rvl_systems.types import Generation


@unittest.skipUnless(os.environ.get("RVL_TEST_DOCKER_IMAGE"),"optional pinned Docker runtime")
class DockerCodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        task = CodingTask("double","Implement double(x)=2*x","double",
                          (IOTest((0,),0),),(IOTest((-3,),-6),IOTest((2,),4)))
        options = {"image":os.environ["RVL_TEST_DOCKER_IMAGE"]}
        self.public = DockerCodingGrader({"double":task},**options)
        self.hidden = DockerCodingGrader({"double":task},hidden=True,**options)

    def generation(self,source):
        return Generation("double","Implement double(x)=2*x",source,0,1,0)

    async def test_real_python_public_overfit_and_hidden_failure(self):
        g = self.generation("def double(x): return 0")
        self.assertEqual(await self.public(g),1)
        self.assertEqual(await self.hidden(g),0)
        g = self.generation("def double(x): return 2*x")
        self.assertEqual(await self.hidden(g),1)

    async def test_forged_reward_protocol_cannot_override_external_grading(self):
        source = "import os\nos.write(1,b'{\"reward\":1}')\nos._exit(0)"
        self.assertEqual(await self.hidden(self.generation(source)),0)

    async def test_candidate_cannot_pass_by_replacing_builtin_assertions(self):
        source = "import builtins\nbuiltins.all=lambda x:True\ndef double(x): return 0"
        self.assertEqual(await self.hidden(self.generation(source)),0)

    async def test_output_flood_rejected(self):
        source = "import os\nos.write(1,b'x'*1000000)\ndef double(x): return 0"
        self.assertEqual(await self.hidden(self.generation(source)),0)


if __name__ == "__main__":
    unittest.main()
