from typing import OrderedDict
from type_enforced import Enforcer
from .interactions import Transition, Decay, Coupling, Rabi
from .energy_level import EnergyLevel

class Component():
    def __init__(self, energy_levels : list[EnergyLevel] , name : str ) -> None:      
        # assert no duplicate energy levels
        if len(energy_levels) != len(set(energy_levels)):
            raise Exception("There are duplicate energy levels")
        assert isinstance(name,str)

        self.name = name
        self.energy_levels = OrderedDict()
        for idx ,energy_level in enumerate(energy_levels):
            assert isinstance(energy_level,EnergyLevel)
            
            energy_level._associate_with_component(self)
            energy_level.index_in_component = idx

            self.energy_levels[energy_level.name] = energy_level
            
        
        self.dimension = len(energy_levels)

        self.rabis = {}
        self.decays = {}
        self.couplings = {}

        self.coupled_components = {}

        self.associated_system = None
        self.index_in_system = None

      
    def __repr__(self) -> str:
        return f'Component {self.name} with {len(self.energy_levels)} energy levels'

    @Enforcer
    def _add_rabi(self, rabi : Rabi) -> None:
        '''
        Adds an interaction with its complex conjugate.
        '''
        if rabi.name in self.rabis:
            raise Exception(f'Rabi {rabi.name} already exists for the component {self.name}')
        else:    
            self.rabis[rabi.name] = rabi


    @Enforcer
    def _add_decay(self, decay : Decay ) -> None:
        '''
        Adds a decay.
        '''
        if decay.name in self.decays:
            raise Exception(f'Decay {decay.name} already exists for the component {self.name}')
        else:    
            self.decays[decay.name] = decay
    

    @Enforcer
    def _add_coupling(self, coupling : Coupling ) -> None:
        '''
        Adds a decay.
        '''
        if coupling.name in self.couplings:
            raise Exception(f'Coupling {coupling.name} already exists for the component {self.name}')
        else:    
            self.couplings[coupling.name] = coupling

    
    def _update_associated_energy_level_excitation_statuses(self) -> None :
        
        for energy_level_name in self.energy_levels:
            # Update excitation status of enegy level
            self.energy_levels[energy_level_name]._update_excitation_status()

    
    def _update_coupled_components(self) -> None:
        # Access all couplings
        for coupling in self.couplings.values():
            # Access the two components of the coupling
            for component in [coupling.associated_component_a, coupling.associated_component_b]:
                # If it is not itself and it does not already exist in the list
                if (component != self ) and ( component.name not in self.coupled_components):
                    # Add to the dictionary of coupled components
                    self.coupled_components[component.name] = component
    
    @Enforcer
    def _associate_with_q_op_system(self, q_op_system, index_in_system : int) -> None:
        from .q_ops_system import QOpsSystem
        assert isinstance(q_op_system,QOpsSystem)
        assert index_in_system >= 0

        self.associated_system = q_op_system
        self.index_in_system = index_in_system



    # @Enforcer
    # def delete_rabi(self, rabi : Rabi) -> None:
    #     try:    
    #         del self.rabis[rabi.name]
    #     except KeyError:
    #         raise Exception(f'Rabi {rabi.name} does not exist in the component {self.name}')

    # @Enforcer
    # def delete_coupling(self, coupling : Coupling) -> None:
    #     try:    
    #         del self.couplings[coupling.name]
    #     except KeyError:
    #         raise Exception(f'Coupling {coupling.name} does not exist in the component {self.name}')
    # @Enforcer
    # def delete_rabi(self, decay : Decay) -> None:
    #     try:    
    #         del self.decays[decay.name]
    #     except KeyError:
    #         raise Exception(f'Decay {decay.name} does not exist in the component {self.name}')
