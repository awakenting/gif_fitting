from .GIF import GIF
from .Filter_Powerlaw import Filter_Powerlaw

class GIF_pow(GIF) :
    """
    Generalized Integrate and Fire model with powerlaw basis functions.
    
    See GIF for full description.
    """

    def __init__(self, dt=0.1):
                   
        super(GIF, self).__init__(dt)
        
        self.eta       = Filter_Powerlaw()
        self.gamma     = Filter_Powerlaw()


    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################        
    def initialize_eta(self):
        pass
