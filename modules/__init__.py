import torchvision.models as models

from modules.SparseDenseNetRefinementMask import SparseDenseNetRefinementMask



def get_model(**params):
    model = _get_model_instance(params['name'])
    
    if params['name'].lower() in ["sparsedensenetrefinementmask"] :
            model = model(max_disp=params['max_disp'], base_channels=params['base_channels'], cost_func=params['cost_func'],
                          num_stage=params['num_stage'], down_scale=params['down_scale'],
                          step=params['step'], samp_num=params['samp_num'], sample_spa_size_list=params['sample_spa_size_list'],
                          down_func_name=params['down_func_name'], weights=params['weights'], grad_method=params['grad_method'], if_overmask=params['if_overmask'], skip_stage_id=params['skip_stage_id'], use_detail=params['use_detail'], thold=params["thold"])

    else :
        raise Exception("No such model: {}".format(params['name']))
    
    return model


def _get_model_instance(name):
    try:
        return {
            'sparsedensenetrefinementmask': SparseDenseNetRefinementMask,
        }[name.lower()]
    except:
        print('Model {} not available'.format(name))
