Here is a point-wise differentiation between **Semantic Network** and **Frame System**:

### **Semantic Network**
1. **Definition**: A semantic network is a graphical representation of knowledge where concepts are represented as nodes, and relationships between concepts are represented as edges.
2. **Structure**: It consists of nodes (representing entities) connected by arcs (representing relationships).
3. **Representation**: It focuses on the relationships between concepts, often using predicates or labels on the arcs to define these relationships.
4. **Type of Knowledge**: Primarily represents **associative knowledge** or relationships between concepts (e.g., "is-a," "has-a").
5. **Inheritance**: Inheritance in semantic networks is typically represented by hierarchical structures where more general concepts are linked to more specific concepts.
6. **Flexibility**: Flexible in terms of adding or modifying relationships between entities.
7. **Usage**: Often used in natural language processing, knowledge representation, and artificial intelligence to model relationships in a simple and intuitive way.
8. **Example**: A node for "Dog" might be connected to a node for "Animal" with an "is-a" relationship.

### **Frame System**
1. **Definition**: A frame system is a data structure for representing stereotypical knowledge or situations, where a **frame** represents a situation or object, and it contains slots for various attributes and values.
2. **Structure**: Frames are complex structures containing a variety of slots, each associated with a set of values or subframes (subframes can represent detailed information).
3. **Representation**: Frames represent both **attributes and relationships** of entities, offering a more detailed and organized structure compared to semantic networks.
4. **Type of Knowledge**: Represents **structured knowledge** with predefined slots for attributes and specific values.
5. **Inheritance**: Inheritance is more structured, where child frames inherit attributes and relationships from parent frames (like object-oriented inheritance).
6. **Flexibility**: Less flexible than semantic networks due to predefined slots and values that represent specific knowledge types.
7. **Usage**: Often used in AI systems for modeling real-world objects, human-like reasoning, and decision-making. It is more suitable for representing complex structures or detailed scenarios.
8. **Example**: A frame for "Car" may have slots such as "color," "make," "model," "engine type," each with specific values.

### **Key Differences:**
| **Aspect**              | **Semantic Network**                            | **Frame System**                        |
|-------------------------|-------------------------------------------------|-----------------------------------------|
| **Concept Representation** | Concepts as nodes, relationships as arcs.      | Knowledge in predefined slots within frames. |
| **Granularity**         | Generally simpler, only shows relationships.   | More detailed, representing attributes and values. |
| **Flexibility**         | More flexible in adding relationships.          | More rigid with predefined slots and attributes. |
| **Inheritance**         | Inheritance through hierarchical relationships. | Inheritance through frames (parent-child). |
| **Type of Knowledge**   | Represents associative knowledge.              | Represents structured, contextual knowledge. |
| **Usage**               | Simpler tasks like concept mapping.            | Complex tasks requiring detailed object representation. |

In summary, a **semantic network** is more focused on representing the relationships between concepts, while a **frame system** is better suited for representing detailed, structured knowledge about specific objects or situations.