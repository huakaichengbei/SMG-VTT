from smgvtt.core.config import ModelConfig
from smgvtt.encoders.tactile.graphs.dilation_schedule import tactile_k_schedule,tactile_dilation_schedule
if __name__=='__main__':
    c=ModelConfig()
    print('TiG k:',tactile_k_schedule(c.tactile.layers,c.tactile.k_start,c.tactile.k_end))
    print('TiG dilation:',tactile_dilation_schedule(c.tactile.layers))
    print('visual layers:',c.visual.layers,'SGPM dim:',c.sgpm.embedding_dim)
