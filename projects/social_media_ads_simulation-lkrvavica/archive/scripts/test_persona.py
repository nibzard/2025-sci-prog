from world.definitions import User

u = User(1, 'm', 25, "['writer']", "['reading']", 'y')
print('Persona loaded:', u.persona_narrative is not None)
if u.persona_narrative:
    print('Word count:', len(u.persona_narrative.split()))
    print('First 100 chars:', u.persona_narrative[:100])
