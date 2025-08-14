class Student:
    """
    Classe pour étudiants prenant l'âge, le score moyen, le nombre d'absences 
    et la volonté de poursuivre aux cycles supérieurs. 
    Comprend un getter pour chaque attribut et une fonction de présentation.
    """
    def __init__(self, age, gmoy, absences, higher):
        self.age = age
        self.gmoy = gmoy
        self.absences = absences
        self.higher = higher

    def get_age(self): # Retourne l'âge
        return self.age
    
    def get_gmoy(self): # Retourne le résultat moyen
        return self.gmoy
    
    def get_absences(self): # Retourne le nombre d'absences
        return self.absences
    
    def get_higher(self): # Retourne la volonté de poursuivre études sup.
        return self.higher
    
    def presentation(self):
        print("\nBonjour! Je suis un étudiant de {:.0f} ans.".format(self.age)) 
        print("J'ai un résultat moyen de {:.2f} et {:.0f} absences.".format(
              self.gmoy, self.absences))
        print("Je prévois poursuivre aux cycles supérieurs." 
              if self.higher == 1 else 
              "Je ne prévois pas poursuivre aux cycles supérieurs.")

