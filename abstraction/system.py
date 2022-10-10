from type_enforced import Enforcer
from .component import Component

class System():

    def __init__(self, name : str, components : list[Component] ) -> None:
        assert isinstance(name,str)
        for component in components:
            assert isinstance(component, Component)
        
        self.name = name
        self.components = {}
        for component in components:
            self.components[component.name] = component
        
        self.compile()

    def __repr__(self) -> str:
        return f'System {self.name} with {len(self.components)} components' 

    @Enforcer
    def add_component(self, component : Component):
        self.components[component.name] = component
        self.compiled = False

    @Enforcer
    def delete_component(self, component : Component):
        del self.components[component.name] 
        self.compiled = False
        

    def compile(self) -> None :
        new_components = {}
        # Access all components
        for component in self.components.values():
            # Update coupled component list
            component.update_coupled_components()
            for coupled_component_name, coupled_component in component.coupled_components.items():
                # Add the components to the list
                new_components[coupled_component_name] = coupled_component
        
        self.components.update(new_components)
        self.compiled = True