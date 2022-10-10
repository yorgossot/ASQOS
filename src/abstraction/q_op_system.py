from type_enforced import Enforcer
from collections import OrderedDict

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
        

    def compile(self) -> None :
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

        # Make dimension list which correspons to the Ordered Dict
        self.dimensions = []
        for position_in_system, component in enumerate(self.components.values()):
            # Obtain the dimension of the component
            self.dimensions.append(component.dimension)
            # Associate the component with the system and its position in the system
            component._associate_with_q_op_system(self, position_in_system)

        # Note that it has been compiled
        self.compiled = True
    
    
    def obtain_effective_operators(self):
        from ..adiabatic_approximation import EffectiveOperatorFormalism
        
        # If not compiled, compile the system
        if not self.compiled:
            self.compile()
        
        self.effective_operator_formalism = EffectiveOperatorFormalism(self)


    # @Enforcer
    # def delete_component(self, component : Component):
    #     del self.components[component.name] 
    #     self.compiled = False
        
        
        
            