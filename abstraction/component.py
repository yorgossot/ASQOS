from type_enforced import Enforcer
from .interactions import Transition, Decay, Coupling
from .energy_level import EnergyLevel

class Component():
    def __init__(self, energy_levels : list[EnergyLevel] , name : str ) -> None:      
        # assert no duplicate energy levels
        if len(energy_levels) != len(set(energy_levels)):
            raise Exception("There are duplicate energy levels")
        assert isinstance(name,str)

        self.name = name
        self.energy_levels = {}
        for energy_level in energy_levels:
            assert isinstance(energy_level,EnergyLevel)
            energy_level.associate_with_component(self)
            self.energy_levels[energy_level.name] = energy_level
             
        self.transitions = {}
        self.decays = {}
        self.couplings = {}

        self.coupled_components = {}

      
    def __repr__(self) -> str:
        return f'Component {self.name} with {len(self.energy_levels)} energy levels'

    @Enforcer
    def add_transition(self, transition : Transition) -> None:
        '''
        Adds an interaction with its complex conjugate.
        '''
        if transition.name in self.transitions:
            raise Exception(f'Decay {transition.name} already exists for the component {self.name}')
        else:    
            self.transitions[transition.name] = transition

    @Enforcer
    def delete_transition(self, transition : Transition) -> None:
        try:    
            del self.transitions[transition.name]
        except KeyError:
            raise Exception(f'Transition {transition.name} does not exist in the component {self.name}')


    @Enforcer
    def add_decay(self, decay : Decay ) -> None:
        '''
        Adds a decay.
        '''
        if decay.name in self.decays:
            raise Exception(f'Decay {decay.name} already exists for the component {self.name}')
        else:    
            self.decays[decay.name] = decay
    

    @Enforcer
    def delete_decay(self, decay : Decay) -> None:
        try:    
            del self.decays[decay.name]
        except KeyError:
            raise Exception(f'Decay {decay.name} does not exist in the component {self.name}')

    @Enforcer
    def add_coupling(self, coupling : Coupling ) -> None:
        '''
        Adds a decay.
        '''
        if coupling.name in self.couplings:
            raise Exception(f'Coupling {coupling.name} already exists for the component {self.name}')
        else:    
            self.couplings[coupling.name] = coupling
    

    @Enforcer
    def delete_coupling(self, coupling : Coupling) -> None:
        try:    
            del self.couplings[coupling.name]
        except KeyError:
            raise Exception(f'Coupling {coupling.name} does not exist in the component {self.name}')

    def update_coupled_components(self) -> None:
        # Access all couplings
        for coupling in self.couplings.values():
            # Access the two components of the coupling
            for component in [coupling.associated_component_a, coupling.associated_component_b]:
                # If it is not itself and it does not already exist in the list
                if (component != self ) and ( component.name not in self.coupled_components):
                    # Add to the dictionary of coupled components
                    self.coupled_components[component.name] = component

            