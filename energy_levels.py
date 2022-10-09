import sympy
from type_enforced import Enforcer

class Component():
    pass

class EnergyLevel():

    def __init__(self, name : str, energy : int = 0 , stable : bool = True ) -> None:
        self.stable = stable
        self.name = name
        self.energy = energy

    def __repr__(self) -> str:
        return f"Energy Level {str(self.name)} with energy E={str(self.energy)}" 

    def set_energy(self, energy ):
        self.energy = energy
    
    def associate_with_component(self, component : Component) -> None:
        assert isinstance(component, Component)
        if not hasattr(self, 'associated_component'):
            self.associated_component = component
        else:
            raise Exception(f"Cannot associate {self} with component {component} since it \
                is already associated with {self.associated_component}")
    
    

class Transition():
    @Enforcer
    def __init__(self, energy_level_ket : EnergyLevel, energy_level_bra : EnergyLevel, coefficient) -> None:
        if not (hasattr(energy_level_ket, 'associated_component') and hasattr(energy_level_bra, 'associated_component')):
            raise Exception('Both energy levels have to be associated with a component.')
        if energy_level_ket.associated_component != energy_level_bra.associated_component:
            raise Exception('Both energy levels have to be associated with the same component.')
        
        self.energy_level_ket = energy_level_ket
        self.energy_level_bra = energy_level_bra
        self.coefficient = sympy.sympify(coefficient)
        
        self.name = energy_level_ket.name + ' <-> ' + energy_level_bra.name

        self.associated_component = energy_level_bra.associated_component
        self.associated_component.add_transition(self)

    def __repr__(self) -> str:
        return f"Transition {self.name} with coefficient {str(self.coefficient)} for component {self.associated_component.name}" 

    def __delete__(self):
        print('deleting')
        self.associated_component.delete_transition(self)


class Decay():
    @Enforcer
    def __init__(self, energy_level_ket : EnergyLevel, energy_level_bra : EnergyLevel , coefficient ) -> None:
        if not (hasattr(energy_level_ket, 'associated_component') and hasattr(energy_level_bra, 'associated_component')):
            raise Exception('Both energy levels have to be associated with a component.')
        if energy_level_ket.associated_component != energy_level_bra.associated_component:
            raise Exception('Both energy levels have to be associated with the same component.')
        if coefficient < 0 :
            raise Exception('Loss rate has to be non-negative') 
        
        self.energy_level_ket = energy_level_ket
        self.energy_level_bra = energy_level_bra
        self.coefficient = sympy.sympify(coefficient)

        self.name =  energy_level_ket.name + ' <~ ' + energy_level_bra.name 

        self.associated_component = energy_level_bra.associated_component
        self.associated_component.add_decay(self)
    
    def __repr__(self) -> str:
        return f"Decay {self.name} with coefficient {str(self.coefficient)} for component {self.associated_component.name}" 
    
    def __del__(self):
        self.associated_component.delete_decay(self)



class Coupling():
    def __init__(self, transition_a : Transition , transition_b : Transition, coefficient ) -> None:
        self.transition_a = transition_a
        self.associated_component_a = transition_a.associated_component
        
        self.transition_b = transition_b
        self.associated_component_b = transition_b.associated_component

        self.coefficient = sympy.sympify(coefficient)




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

      
    def __repr__(self) -> str:
        return str(self.name) + f' with {len(self.energy_levels)} energy levels'

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
    def add_decay(self, decay : Decay ) -> None:
        '''
        Adds a decay.
        '''
        if decay.name in self.decays:
            raise Exception(f'Decay {decay.name} already exists for the component {self.name}')
        else:    
            self.decays[decay.name] = decay
    
    @Enforcer
    def delete_transition(self, transition : Transition) -> None:
        try:    
            del self.transitions[transition.name]
        except KeyError:
            raise Exception(f'Transition {transition.name} does not exist in the component {self.name}')

    @Enforcer
    def delete_decay(self, decay : Decay) -> None:
        try:    
            del self.decays[decay.name]
        except KeyError:
            raise Exception(f'Decay {decay.name} does not exist in the component {self.name}')

