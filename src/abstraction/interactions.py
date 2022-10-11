from .energy_level import EnergyLevel
from type_enforced import Enforcer
import sympy


class Transition():
    @Enforcer
    def __init__(self, energy_level_ket : EnergyLevel, energy_level_bra : EnergyLevel) -> None:
        if not (hasattr(energy_level_ket, 'associated_component') and hasattr(energy_level_bra, 'associated_component')):
            raise Exception('Both energy levels have to be associated with a component.')
        if energy_level_ket.associated_component != energy_level_bra.associated_component:
            raise Exception('Both energy levels have to be associated with the same component.')
        
        self.energy_level_ket = energy_level_ket
        self.energy_level_bra = energy_level_bra
        
        self.name = '|'+ energy_level_ket.name + '><' + energy_level_bra.name +'|'

        self.associated_component = energy_level_bra.associated_component
    

class Rabi(Transition):
    '''
    Rabi transition. Give as energy_level_ket the excited state.

    TODO: Implement classification of Rabi transitions into exciting.
    '''
    @Enforcer
    def __init__(self, energy_level_ket: EnergyLevel, energy_level_bra: EnergyLevel, coefficient) -> None:
        super().__init__(energy_level_ket, energy_level_bra)
        
        self.coefficient = sympy.sympify(coefficient)

        # Update component and excited state
        self.energy_level_ket._add_rabi_energy_level(self.energy_level_bra)
        self.energy_level_bra._add_rabi_energy_level(self.energy_level_ket)

        self.associated_component._add_rabi(self)
    
    def __repr__(self) -> str:
        return f"Decay: {self.name} , Coefficient: {str(self.coefficient)} , Associated component: {self.associated_component.name}" 
    


class Decay(Transition):
    '''
    coefficient will be taken as sqrt
    '''
    @Enforcer
    def __init__(self, excited_energy_level : EnergyLevel, decayed_energy_level : EnergyLevel , coefficient ) -> None:
        super().__init__( energy_level_ket = decayed_energy_level, energy_level_bra = excited_energy_level)
        
        if coefficient < 0 :
            raise Exception('Loss rate has to be non-negative') 
        self.coefficient = sympy.sqrt(sympy.sympify(coefficient))

        self.excited_energy_level = excited_energy_level
        self.decayed_energy_level = decayed_energy_level

        # Update component and excited state
        self.excited_energy_level._add_decayed_energy_level(self.decayed_energy_level)
        self.decayed_energy_level._add_decaying_energy_level(self.excited_energy_level)
        self.associated_component._add_decay(self)
    
    def __repr__(self) -> str:
        return f"Decay: {self.name} , Coefficient: {str(self.coefficient)} , Associated component: {self.associated_component.name}" 
    
    # def delete(self) -> None:
    #     self.associated_component.delete_rabi(self)
    #     self.excited_energy_level.delete_decayed_energy_level(self.decayed_energy_level)



class Coupling():
    @Enforcer
    def __init__(self, transition_a : Transition , transition_b : Transition, coefficient ) -> None:
        self.transition_a = transition_a
        self.associated_component_a = transition_a.associated_component
        
        self.transition_b = transition_b
        self.associated_component_b = transition_b.associated_component

        self.coefficient = sympy.sympify(coefficient)

        self.name = transition_a.name + transition_b.name

        self.associated_component_a._add_coupling(self)        
        self.associated_component_b._add_coupling(self)

        for transition in [self.transition_a, self.transition_b]: 
            transition.energy_level_bra._add_coupled_energy_level(transition.energy_level_ket)
            transition.energy_level_ket._add_coupled_energy_level(transition.energy_level_bra)

 
    def __repr__(self) -> str:
        return f"Coupling {self.name} with coefficient: {self.coefficient}"

    # def delete(self) -> None: 
    #     # Update components
    #     self.associated_component_a.delete_coupling(self)
    #     self.associated_component_b.delete_coupling(self)
    #     # Update energy levels
    #     for associated_component in [self.associated_component_a, self.associated_component_b]:
    #         energy_level_1 , energy_level_2 = associated_component.energy_levels.values()
    #         energy_level_1.add_coupled_energy_level(energy_level_2)
    #         energy_level_2.add_coupled_energy_level(energy_level_1)