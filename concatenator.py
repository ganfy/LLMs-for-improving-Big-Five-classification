# Definir nombre de archivo de entrada y salida
input_file = 'conversation.txt'
output_file = 'RAGs.csv'

# Leer el contenido del archivo de entrada
with open(input_file, 'r') as f:
    lines = f.readlines()

# Inicializar variables
responses = []
labels = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
current_response = ''
current_labels = []

# Procesar cada línea del archivo
i = 0
while i < len(lines):

    line = lines[i].strip()
    # Identificar inicio de sección del cliente
    if line.startswith('Client:'):
        current_response += line[8:] + ' '  # Eliminar "Client: " al inicio
        i += 1
    elif line.startswith('Psychotherapist:'):
        i += 1  # Saltar línea de Psychotherapist
        
        # Leer etiquetas de la siguiente línea
    elif line != '':
        current_labels = lines[i].strip().split(", ")
        print(current_labels)
        i += 2  # Saltar línea en blanco después de las etiquetas
        
        # Guardar la respuesta del cliente y etiquetas encontradas
        if current_labels:
            responses.append((current_response.strip(), *current_labels))
            print(len(responses[-1]))
            current_response = ''
            current_labels = []
    else:
        i += 1

# Escribir las respuestas y etiquetas en el archivo CSV
with open(output_file, 'w') as f:
    f.write('"TEXT","cEXT","cNEU","cAGR","cCON","cOPN"\n')
    for response in responses:
        f.write(f'"{response[0]}",{response[1]},{response[2]},{response[3]},{response[4]},{response[5]}\n')

print(f'Se ha generado el archivo {output_file} correctamente.')
