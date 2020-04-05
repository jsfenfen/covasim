import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cvu
from . import person as cvper
from . import parameters as cvpars


class Population(sc.prettyobj):
    """
    Class to represent a population of people

    A population is defined by
    - A collection of people (`Person` instances)
    - A collection of networks specifying how those people interact (a collection of `ContactLayer` instances)

    Thus this class essentially specifies the graph upon which infection and
    transmission take place

    """

    def __init__(self):
        self.people = {}  #: Store Person instances
        self.contact_layers = {}  #: Store ContactLayer instances
        self._uids = {}  #: Map index to UID

    def get_person(self, ind):
        ''' Return a person based on their ID '''
        return self.people[self._uids[ind]]

    @classmethod
    def random(cls, pars, n_people: int = None, n_regular_contacts: int = 20, n_random_contacts: int = 0, id_len=6):
        """
        Make a simple random population

        Args:
            pars: Simulation parameters
            n_people: Number of people in population
            n_infected: Number of seed infections
            n_regular_contacts: Regular/repeat number of contacts (e.g. household size)
            n_random_contacts: Number of random contacts (e.g. community encounters per day)
            id_len: Optionally specify UUID length (may be necessary if requesting a very large number of people)

        Returns: A Population instance

        """

        self = cls()

        if n_people is None:
            n_people = pars['n']

        # Load age data based on 2018 Seattle demographics
        age_data = np.array([
            [0, 4, 0.0605],
            [5, 9, 0.0607],
            [10, 14, 0.0566],
            [15, 19, 0.0557],
            [20, 24, 0.0612],
            [25, 29, 0.0843],
            [30, 34, 0.0848],
            [35, 39, 0.0764],
            [40, 44, 0.0697],
            [45, 49, 0.0701],
            [50, 54, 0.0681],
            [55, 59, 0.0653],
            [60, 64, 0.0591],
            [65, 69, 0.0453],
            [70, 74, 0.0312],
            [75, 79, 0.02016],  # Calculated based on 0.0504 total for >=75
            [80, 84, 0.01344],
            [85, 89, 0.01008],
            [90, 99, 0.00672],
        ])

        # Handle sex and UID
        uids = sc.uuid(which='ascii', n=n_people, length=id_len, tostring=True)
        sexes = cvu.rbt(0.5, n_people)

        # Handle ages
        age_data_min = age_data[:, 0]
        age_data_max = age_data[:, 1] + 1  # Since actually e.g. 69.999
        age_data_range = age_data_max - age_data_min
        age_data_prob = age_data[:, 2]
        age_data_prob /= age_data_prob.sum()  # Ensure it sums to 1
        age_bins = cvu.mt(age_data_prob, n_people)  # Choose age bins
        ages = age_data_min[age_bins] + age_data_range[age_bins] * np.random.random(n_people)  # Uniformly distribute within this age bin

        # Instantiate people
        self.people = {uid: cvper.Person(pars=pars, uid=uid, age=age, sex=sex) for uid, age, sex in zip(uids, ages, sexes)}
        self._uids = {i: x.uid for i, x in enumerate(self.people.values())}

        # Make contacts
        self.contact_layers = {}

        # Make static contact matrix
        contacts = {}
        for i, person in enumerate(self.people.values()):
            n_contacts = cvu.pt(n_regular_contacts)  # Draw the number of Poisson contacts for this person
            contacts[person.uid] = cvu.choose(max_n=n_people, n=n_contacts)  # Choose people at random, assigning to 'household'
        layer = StaticContactLayer(name='Household', contacts=contacts)
        self.contact_layers[layer.name] = layer

        # Make random contacts
        if n_random_contacts > 0:
            self.contact_layers['Community'] = RandomContactLayer(name='Community', max_n=n_people, n=n_random_contacts)

        return self

    @classmethod
    def synthpops(cls, pars, n_people=5000, n_random_contacts: int = 20, betas=None):
        """
        Construct network with microstructure using Synthpops

        Args:
            pars: Covasim parameters (e.g. output from `covasim.make_pars()`) used when initializing people
            n_people: Number of people
            n_random_contacts: Number of random community contacts each day
            beta: Baseline beta value
            betas: Optionally specify dict with relative beta values for each contact layer

        Returns: A Population instance

        """

        if betas is None:
            betas = {'H': 1.7, 'S': 0.8, 'W': 0.8, 'R': 0.3}  # Per-population beta weights; relative

        import synthpops as sp  # Optional import
        population = sp.make_population(n_people)

        self = cls()

        # Make people
        self.people = {}
        for uid, person in population.items():
            self.people[uid] = Person(pars=pars, uid=uid, age=person['age'], sex=person['sex'])
        self._uids = {i: x.uid for i, x in enumerate(self.people.values())}

        # Make contact layers
        layers = ['H', 'S', 'W']  # Hardcode the expected synthpops contact layers for now
        self.contact_layers = {}
        uid_to_index = {x.uid: i for i, x in enumerate(self.people.values())}
        for layer in layers:
            contacts = {}
            for uid, person in population.items():
                contacts[uid] = np.array([uid_to_index[uid] for uid in person['contacts'][layer]], dtype=np.int64)  # match datatype in covasim.utils.bf
                self.people[uid] = Person(pars=pars, uid=uid, age=person['age'], sex=person['sex'])
            self.contact_layers[layer] = StaticContactLayer(name=layer, beta=betas[layer], contacts=contacts)
        self.contact_layers['R'] = RandomContactLayer(name='R', beta=betas['R'], max_n=n_people, n=n_random_contacts)

        return self

    @classmethod
    def country(cls, country_code, beta=0.015):
        """
        Create population from country data

        Args:
            country_code: ISO Country code to specify country e.g. 'IND', 'TZA'

        Returns: A Population instance

        """
        raise NotImplementedError

    @staticmethod
    def load(filename, *args, **kwargs):
        '''
        Load the population dictionary from file.

        Args:
            filename (str): name of the file to load.
        '''
        filepath = sc.makefilepath(filename=filename, *args, **kwargs)
        pop = sc.loadobj(filepath)
        if not isinstance(pop, Population):
            raise TypeError('Loaded file was not a population')
        return pop

    def save(self, filename, *args, **kwargs):
        '''
        Save the population dictionary to file.

        Args:
            filename (str): name of the file to save to.
        '''
        filepath = sc.makefilepath(filename=filename, *args, **kwargs)
        sc.saveobj(filepath, self)
        return filepath


class ContactLayer(sc.prettyobj):
    """

    Beta is stored as a single scalar value so that it can be overwritten or otherwise
    modified by interventions in a consistent fashion

    """

    def __init__(self, name: str, beta: float, traceable: bool = True) -> None:
        self.name = name  #: Name of the contact layer e.g. 'Households'
        self.beta = beta  #: Transmission probability per contact (absolute)
        self.traceable = traceable  #: If True, the contacts should be considered tracable via contact tracing
        return

    def get_contacts(self, person, sim) -> list:
        """
        Get contacts for a person

        :param person:
        :param sim: The simulation instance
        :return: List of contact *indexes* e.g. [1,50,295]

        """
        raise NotImplementedError


class StaticContactLayer(ContactLayer):
    def __init__(self, name: str, contacts: dict, beta: float = 1.0) -> None:
        """
        Contacts that are the same every timestep

        Suitable for groups of people that do not change over time e.g., households, workplaces

        Args:
            name:
            beta:
            contacts:

        """

        super().__init__(name, beta)
        self.contacts = contacts  #: Dictionary mapping `{source UID:[target indexes]}` storing interactions
        return

    def get_contacts(self, person, sim) -> list:
        return self.contacts[person.uid]


class RandomContactLayer(ContactLayer):
    def __init__(self, name: str, max_n: int, n: int, beta: float = 1.0) -> None:
        """
        Randomly sampled contacts each timestep

        Suitable for interactions that randomly occur e.g., community transmission

        Args:
            name:
            beta: Transmission probability per contact (absolute)
            max_n: Number of people available
            n: Number of contacts per person

        """

        super().__init__(name, beta, traceable=False)  # nb. cannot trace random contacts e.g. in community
        self.max_n = max_n
        self.n = n  #: Number of randomly sampled contacts per timestep

    def get_contacts(self, person, sim) -> list:
        return cvu.choose(max_n=self.max_n, n=self.n)


def set_prognoses(sim, popdict):
    '''
    Determine the prognosis of an infected person: probability of being aymptomatic, or if symptoms develop, probability
    of developing severe symptoms and dying, based on their age
    '''

    # Initialize input and output
    by_age = sim['prog_by_age']
    ages = sc.promotetoarray(popdict['age']) # Ensure it's an array
    n = len(ages)
    prognoses = sc.objdict()

    prog_pars = cvpars.get_default_prognoses(by_age=by_age)

    # If not by age, same value for everyone
    if not by_age:

        prognoses.symp_prob   = sim['rel_symp_prob']   * prog_pars.symp_prob   * np.ones(n)
        prognoses.severe_prob = sim['rel_severe_prob'] * prog_pars.severe_prob * np.ones(n)
        prognoses.crit_prob   = sim['rel_crit_prob']   * prog_pars.crit_prob   * np.ones(n)
        prognoses.death_prob  = sim['rel_death_prob']  * prog_pars.death_prob  * np.ones(n)

    # Otherwise, calculate probabilities of symptoms, severe symptoms, and death by age
    else:
        # Conditional probabilities of severe symptoms (given symptomatic) and death (given severe symptoms)
        severe_if_sym   = np.array([sev/sym  if sym>0 and sev/sym>0  else 0 for (sev,sym)  in zip(prog_pars.severe_probs, prog_pars.symp_probs)]) # Conditional probabilty of developing severe symptoms, given symptomatic
        crit_if_severe  = np.array([crit/sev if sev>0 and crit/sev>0 else 0 for (crit,sev) in zip(prog_pars.crit_probs,   prog_pars.severe_probs)]) # Conditional probabilty of developing critical symptoms, given severe
        death_if_crit   = np.array([d/c      if c>0   and d/c>0      else 0 for (d,c)      in zip(prog_pars.death_probs,  prog_pars.crit_probs)])  # Conditional probabilty of dying, given critical

        symp_probs     = sim['rel_symp_prob']   * prog_pars.symp_probs  # Overall probability of developing symptoms
        severe_if_sym  = sim['rel_severe_prob'] * severe_if_sym         # Overall probability of developing severe symptoms (https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf)
        crit_if_severe = sim['rel_crit_prob']   * crit_if_severe        # Overall probability of developing critical symptoms (derived from https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm)
        death_if_crit  = sim['rel_death_prob']  * death_if_crit         # Overall probability of dying (https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)

        # Calculate prognosis for each person
        symp_prob, severe_prob, crit_prob, death_prob  = [],[],[],[]
        age_cutoffs = prog_pars.age_cutoffs
        for age in ages:
            # Figure out which probability applies to a person of the specified age
            ind = next((ind for ind, val in enumerate([True if age < cutoff else False for cutoff in age_cutoffs]) if val), -1)
            this_symp_prob   = symp_probs[ind]    # Probability of developing symptoms
            this_severe_prob = severe_if_sym[ind] # Probability of developing severe symptoms
            this_crit_prob   = crit_if_severe[ind] # Probability of developing critical symptoms
            this_death_prob  = death_if_crit[ind] # Probability of dying after developing critical symptoms
            symp_prob.append(this_symp_prob)
            severe_prob.append(this_severe_prob)
            crit_prob.append(this_crit_prob)
            death_prob.append(this_death_prob)

        # Return output
        prognoses.symp_prob   = symp_prob
        prognoses.severe_prob = severe_prob
        prognoses.crit_prob   = crit_prob
        prognoses.death_prob  = death_prob

    popdict.update(prognoses) # Add keys to popdict

    return
