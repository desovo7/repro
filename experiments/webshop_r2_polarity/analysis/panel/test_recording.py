"""Regression: recording must not re-project the native mutable action list."""
import numpy as np
from omegaconf import OmegaConf
from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.env_package.webshop import webshop_projection
from recording import make_recorded_manager

TASK='Find me a blue round rug, and price lower than60dollars'


class Backend:
    def reset(self):
        return ['WebShop [SEP] Instruction: [SEP] '+TASK+' [SEP] Search'], [dict(available_actions={'has_search_bar':True,'clickables':['search']},panel_initial_signature='test')]
    def step(self,actions):
        self.last_actions=list(actions)
        return ['WebShop [SEP] Instruction: [SEP] '+TASK+' [SEP] Results'], [0.0],[False],[dict(available_actions={'has_search_bar':False,'clickables':['b012345678']},won=False,task_score=0.0)]


def main():
    config=OmegaConf.create({'env':{'history_length':2}})
    a,b=Backend(),Backend()
    native=WebshopEnvironmentManager(a,webshop_projection,config)
    recorded=make_recorded_manager(WebshopEnvironmentManager)(b,webshop_projection,config)
    native.reset(None);recorded.reset(None)
    messages=['<think>I will search for the requested rug.</think><action>search[blue round rug price lower than60dollars]</action>',
              '<think>Choose this product.</think><action>click[b012345678]</action>',
              'malformed action with no XML tags and a long suffix']
    for i,text in enumerate(messages):
        x=native.step([text]);y=recorded.step([text])
        assert a.last_actions==b.last_actions
        assert x[0]==y[0]
        np.testing.assert_array_equal(x[1],y[1]);np.testing.assert_array_equal(x[2],y[2])
        assert bool(x[3][0]['is_action_valid'])==bool(y[3][0]['is_action_valid'])
        row=recorded.trace_steps[0][-1]
        assert row['generated_text']==text and row['action']==a.last_actions[0]
        assert row['format_valid']==(i<2)
    assert recorded.trace_steps[0][0]['action']=='search[blue round rug price lower than60dollars]'
    print('PASS:recording equals native stepping for full search, product click and malformed output; original generated text preserved.')


if __name__=='__main__':
    main()
