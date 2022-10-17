from type_enforced import Enforcer
from collections import OrderedDict
import qutip 
from numpy import prod
from .. import q_ops_utilities
from src.abstraction.interactions import Rabi

from .component import Component


class QOpsSystem():

    def __init__(self, name : str, components : list[Component] ) -> None:
        assert isinstance(name,str)
        for component in components:
            assert isinstance(component, Component)
        
        self.name = name
        self.components = OrderedDict()
        for component in components:
            self.components[component.name] = component
        
        self.compile()

    def __repr__(self) -> str:
        return f'System {self.name} with {len(self.components)} components' 

    @Enforcer
    def add_component(self, component : Component):
        self.components[component.name] = component
        self.compiled = False
        

    def _update_component_list(self) -> None :
        '''
        Updates list of components by taking into consideration couplings.
        '''
        new_components = OrderedDict()
        # Access all components
        for component in self.components.values():
            # Update coupled component list
            component._update_coupled_components()
            for coupled_component_name, coupled_component in component.coupled_components.items():
                # Add the components to the list
                new_components[coupled_component_name] = coupled_component
        # Update the component dictionary
        self.components.update(new_components)

    def _update_dimensions(self):
        '''
        Updates dimensions of the object with regard to the components.
        '''
        self.dimensions = []
        for  component in self.components.values():
            # Obtain the dimension of the component
            self.dimensions.append(component.dimension)
        self.dimension = prod(self.dimensions)


    def _collect_interactions(self):
        '''
        Collects all interactions, decays and energy levels.
        '''
        self.rabis = {}
        self.couplings = {}
        self.decays = {}
        self.energy_levels = {}

        for component in self.components.values():
            # Collect interactions
            self.rabis.update(component.rabis)
            self.decays.update(component.decays)
            self.couplings.update(component.couplings)
            self.energy_levels.update(component.energy_levels)


    def _generate_hamiltonians(self) -> None :
        '''
        Generates the Hamiltonian elements from the interactions and the energy levels
        '''

        self.hamiltonian = {'rabis': {}, 'couplings' : {}, 'energy_levels': {}}
        
        # Energy Levels
        for energy_level in self.energy_levels.values():
            # Obtain necessary indices and dimension of associated component
            index_in_system = energy_level.associated_component.index_in_system
            idx_in_component = energy_level.index_in_component
            component_dimension = energy_level.associated_component.dimension        
            
            ket = qutip.basis(component_dimension, idx_in_component)
            ketbra = ket.proj()

            interaction_Qobj = q_ops_utilities.embed_ketbras_in_system(ketbra, index_in_system, self.dimensions)
            
            self.hamiltonian['energy_levels'][energy_level.name] = {'Qobj' : interaction_Qobj , 'coefficient' : energy_level.energy}  
        
        # Rabis
        for rabi in self.rabis.values():
            # Obtain necessary indices and dimension of associated component
            index_in_system = rabi.associated_component.index_in_system
            ket_idx_in_component = rabi.energy_level_ket.index_in_component
            bra_idx_in_component = rabi.energy_level_bra.index_in_component
            component_dimension = rabi.associated_component.dimension

            ket = qutip.basis(component_dimension, ket_idx_in_component)
            bra = qutip.basis(component_dimension, bra_idx_in_component).dag()
            ketbra = ket * bra
            
            interaction_Qobj = q_ops_utilities.embed_ketbras_in_system(ketbra, index_in_system, self.dimensions)
            
            self.hamiltonian['rabis'][rabi.name] = {'Qobj' : interaction_Qobj , 
                                                    'coefficient' : rabi.coefficient,
                                                    'interaction_object' : rabi}

        # Couplings
        for coupling in self.couplings.values():
            # Obtain associated components
            associated_transition_a = coupling.transition_a
            associated_transition_b = coupling.transition_b
            
            ketbras = []
            indices_in_system = []
            for transition in [associated_transition_a, associated_transition_b]:
                associated_component = transition.associated_component 
                index_in_system = associated_component.index_in_system
                ket_idx_in_component = transition.energy_level_ket.index_in_component
                bra_idx_in_component = transition.energy_level_bra.index_in_component
                component_dimension = associated_component.dimension
                
                ket = qutip.basis(component_dimension, ket_idx_in_component)
                bra = qutip.basis(component_dimension, bra_idx_in_component).dag()
                ketbra = ket * bra
                
                ketbras.append(ketbra)
                indices_in_system.append(index_in_system)
            
            interaction_Qobj = q_ops_utilities.embed_ketbras_in_system(ketbras, indices_in_system, self.dimensions)
            
            self.hamiltonian['couplings'][coupling.name] = {'Qobj' : interaction_Qobj , 'coefficient' : coupling.coefficient}   


    def _generate_lindblads(self) -> None :
        '''
        Generates the Lindblad operators from the decays.
        '''
        self.lindblads = {}

        for decay in self.decays.values():
            index_in_system = decay.associated_component.index_in_system
            ket_idx_in_component = decay.energy_level_ket.index_in_component
            bra_idx_in_component = decay.energy_level_bra.index_in_component
            component_dimension = decay.associated_component.dimension
        
            ket = qutip.basis(component_dimension, ket_idx_in_component)
            bra = qutip.basis(component_dimension, bra_idx_in_component).dag()
            ketbra = ket * bra
            
            interaction_Qobj = q_ops_utilities.embed_ketbras_in_system(ketbra, index_in_system, self.dimensions)
            
            self.lindblads[decay.name] = {'Qobj' : interaction_Qobj , 'coefficient' : decay.coefficient}
    
    
    def compile(self) -> None :
        '''
        Compiles the object into a final form.
        '''
        self._update_component_list()
        self._update_dimensions()
        self._collect_interactions()
        # Update position in system / component system association
        # and energy level excitation statuses
        for index_in_system, component in enumerate(self.components.values()):
            component._associate_with_q_op_system(self, index_in_system)
            component._update_associated_energy_level_excitation_statuses()
        
        self._generate_hamiltonians()
        self._generate_lindblads()
        # Note that it has been compiled
        self.compiled = True
    
    
    def obtain_effective_operators(self):
        '''
        Make use of the effective operator formalism to obtain effective operators 
        when all rabis are weak.
        '''
        from ..adiabatic_elimination import EffectiveOperatorFormalism
        
        # If not compiled, compile the system
        if not self.compiled:
            self.compile()
        
        self.effective_operator_formalism = EffectiveOperatorFormalism(self)
        



    # @Enforcer
    # def delete_component(self, component : Component):
    #     del self.components[component.name] 
    #     self.compiled = False
        
        
        
            