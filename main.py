name = input("Ingresa tu nombre: ")

title = """ Bienvenido a 
.-..-.   .-.   .-. .-..-.  .-. .-..--. .-..-..-. .-..--. .-..-.   .-..-.   .-..-..-. 
| | ~.-. | |   | | | | ~.-.| | | | ~~   ~ | | ~  | | ~~  | | ~.-. | | ~.-. | | ~ | | 
| |.-.~  | |   | | | |   ~ | | | | _      | |    | | _   | |.-.~  | |.-.~  | |.-.| | 
| | ~.-. | |   | | | |     | | | |`-'     | |    | |`-'  | | ~.-. | | ~.-. | | ~ | | 
| |  | | | | _ | | | |     | | | | __     | |    | | __  | |  | | | |  | | | |   | | 
`-'  `-' `-'`-'`-' `-'     `-' `-'`--'    `-'    `-'`--' `-'  `-' `-'  `-' `-'   `-' 
"""

def bienvenida(hero):
    return(f"Bienvenido {hero} a una nueva aventura en el mundo de Arcane")

print(title)

print(bienvenida(name))

want_to_play = input("Quieres jugar? Si/No ").lower()

if want_to_play == "si":
    print("Te sientas alrededor de una mesa de madera astillada, rodeado por el sonido de carcajadas ruidosas y tragos generosos del mejor ron de Bilgewater. En una esquina, una mujer con un llamativo abrigo morado pone los pies sobre la mesa.") 
    print("Y ahí estábamos,") 
    print("dice ella, con los ojos solemnes, " )
    print("yo y mi tripulación, escarbando en ese maldito naufragio. Cuando de repente—¡zas!") 
    print("Mueve su brazo, casi tirando una jarra de cerveza. ")
    print("Algo oscuro vino desde el horizonte. ¡Nunca sentí algo tan aterrador y frío en mi vida, te lo juro! El barco está embrujado, te digo. Nunca debimos sacar ese naufragio de las mareas.")
    print("Una figura rechoncha con ropas oscuras interviene desde unas mesas más allá. ")
    print("¡Vamos Tala, ya sabemos que no debemos creer en tus cuentos exagerados!")
    print("Un grito repentino se eleva por encima de la multitud.")
    print("Te mostrare lo que pasa si no crees en los cuentos de la bucanera")
    print("Los nudillos crujen contra el hueso, y el mismo hombre rechoncho con ropas oscuras sale volando cuando la mujer de morado le estrella el puño en la cara. Uno de los compañeros de bebida del hombre lo ayuda a levantarse del suelo, mientras otro se lanza hacia la mujer. Alguien en la taberna levanta una jarra en señal de aprobación, y una multitud bulliciosa comienza a animar la pelea. ¿Qué haces?")
    
    decision = input("¿A quién apoyas? Tala / Daine / Nada ").lower()

    if decision == "tala":
        print("Logras acercarte a Tala y defenderla de la gresca que se formó.")
        print("Ya me he cansado de ese idiota. Daine siempre fue un estúpido ambicioso. No tiene ni idea del peligro en el que se está metiendo.")
        print("Ese barco no es un simple naufragio. Lo sacamos de las profundidades hace tres noches... y desde entonces, he visto cosas. Sombras. Siempre acechando en el horizonte, moviéndose como si nos estuvieran observando.")
    elif decision == "daine":
        print("Gracias por echarme una mano ahí... esa Tala es una pieza. Siempre inventando historias para asustar a los otros y quedarse con lo mejor.")
        print("Bah, nada más que cuentos de marineros supersticiosos. El Phantom Ire es solo otro barco hundido, con tesoros para quien sea lo suficientemente listo para sacarlos.")
    elif decision == "nada":
        print("Decides no involucrarte en la gresca.")
    else:
        print("Opción inválida. Vuelve a iniciar.")

    print("En un instante, todas las velas que adornan los candelabros de la taberna parpadean y se apagan. La puerta cruje al abrirse, dejando entrar un viento frío. A través de la luz lunar que se filtra por las ventanas sucias, las sombras de la habitación comienzan a alargarse. Las siluetas se deforman, estirándose en extremidades largas, con garras afiladas y bocas abiertas de par en par. Las sombras se elevan desde el suelo por su propia voluntad. En las cuencas donde deberían estar sus ojos, un brillo fantasmagórico de color azul hambriento ilumina el lugar.")
    print("Luego de una intensa lucha con las sombras escuchas a Diana gritar fuertemente: A las barricadas. Te dirijes a la calle y te percatas de que todos van corriendo hacia el punete del carnicero.")

    decision_refugio = input("Ves a tala correr hacia uno de los callojones cercanos y ves a la multitud dirigirse al lado contrario. ¿La sigues o vas con la multitud? Tala / Multitud ").lower()

    if decision_refugio == "tala":
        print("Tala se acerca con una sonrisa astuta y te dice: Ustedes me ayudaron antes, y no me olvido de eso. Sigan conmigo, les llevaré a un lugar donde podrán estar fuera de peligro.")
        print("""La sombra de un barco masivo se proyecta sobre ustedes, tan imponente que parece devorar la luz de la luna. Las velas negras del navío ondean con fuerza en el viento, agitándose como si fueran hechas de pura oscuridad. Desde la proa, que ha sido tallada en la forma de una enorme mandíbula de tiburón, dos cañones gigantes se levantan, apuntando como si estuvieran listos para destrozar cualquier cosa que se interponga en su camino.

            Una serie de luces verdes resplandecen en la proa, brillando como los ojos de una bestia hambrienta. Parecen observar a todos los que se atreven a acercarse, sin piedad, invitando a la duda y al miedo.

            El sonido del viento y el crujido de las maderas antiguas se mezcla con el suave golpeteo del agua contra el casco, creando una atmósfera ominosa, mientras el Dreadway se alza ante ustedes como una criatura viva, un barco de guerra que ha visto y causado más destrucción de la que jamás podrían imaginar""")
    elif decision_refugio == "multitud":
        print("Las calles de Rat Town pronto ceden paso a un puente de piedra...")
        if decision == "tala":
            print("Daine gruñe: Parece que hicieron bien en confiar en ella.")
            print("Si han llegado hasta aquí, es porque saben cómo enfrentarse a las sombras. Vayan al templo, hablen con Fortune. Necesita gente como ustedes, y por lo que vi en Rat Town, seguro que serán bienvenidos.")
        elif decision == "daine" or decision == "nada":
            print("Daine suelta una risa vacía: Nunca pensé que las historias de Tala fueran ciertas...")
            print("Si han llegado hasta aquí, es porque saben cómo enfrentarse a las sombras. Vayan al templo, hablen con Fortune. Necesita gente como ustedes, y por lo que vi en Rat Town, seguro que serán bienvenidos.")

    else:
        print("Opción inválida. Vuelve a iniciar.")
else:
    print("Vuelve más tarde.")
