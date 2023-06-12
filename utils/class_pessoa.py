import random
import string
from datetime import datetime, timedelta

import pandas as pd

# DICIONÁRIO DE PALAVRAS

# TODO: FIXO obs e tipo_h, cargo, aspa, doar e org.
tipo_h = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "AB"}

fator_rh = {0: "A+", 1: "A-", 2: "B+", 3: "B-", 4: "O+", 5: "O-", 6: "AB+", 7: "AB-"}

obs = {
    0: "EXERCE ATIVIDADE REMUNERADA;",
    1: "A;",
    2: "B;",
    3: "C;",
    4: "D;",
    5: "E;",
    6: "F;",
    7: "G;",
    8: "H;",
    9: "I;",
    10: "J;",
    11: "K;",
    12: "L;",
    13: "M;",
    14: "N;",
    15: "O;",
    16: "P;",
    17: "Q;",
    18: "R;",
    19: "S;",
    20: "T;",
    21: "U;",
    22: "V;",
    23: "W;",
    24: "X;",
    25: "Y;",
    26: "Z;",
}

ufs = [
    'AC',
    'AL',
    'AM',
    'AP',
    'BA',
    'CE',
    'DF',
    'ES',
    'GO',
    'MA',
    'MT',
    'MS',
    'MG',
    'PA',
    'PB',
    'PR',
    'PE',
    'PI',
    'RJ',
    'RS',
    'RN',
    'RO',
    'RR',
    'SC',
    'SP',
    'SE',
    'TO'
]

cargo = {0: "DIRETOR", 1: "COORDENADOR", 2: "PRESIDENTE"}

aspa = {0: "COM AVRB VIUVEZ", 1: "COM AVRB DIVÓRCIO"}

org = {
    0: "SDS",
    1: "SSP",
    2: "POM",
    3: "SNJ",
    4: "SPTC",
    5: "SESP",
    6: "SJS",
    7: "POF",
    8: "SES",
    9: "SEJ",
}
# ------------------------------------------------------------------------------------------------------------------

name_dct_path = "./files/nnomes.txt"

name_dct = []
with open(name_dct_path, "r") as fd:
    for line in fd:
        splt = line.strip("\n").split(" ")
        name_dct += [splt[0] + " "]*int(splt[1])

n_names = len(name_dct)

surname_dct_path = './files/s_nome.txt'
surname_dct = []
with open(surname_dct_path, "r") as fd:
    for line in fd:
        surname_dct += [line.strip("\n")]

n_surnames = len(surname_dct)

df = pd.read_csv(r"./files/cid_est.csv", encoding="utf-8")


def gen_rand_datetime(min_time=datetime(1950,1,1,00,00,00), max_time=datetime.now()):
    # Get total number of days between min and max time
    total_days = (max_time - min_time).days
    
    # Generate random number of days in between
    rand_days = random.randint(1, total_days-1)

    # add rand_days to min date, generating rand date between min and max
    new_time = min_time + timedelta(days=rand_days)
    return new_time

def time_to_str(timestr):
    ret = timestr.strftime("%d/%m/%Y")
    return ret

def make_rg():
    def formata_rg(n_rg):
        rg_f = ""
        for i in range(len(n_rg)):
            rg_f = rg_f + n_rg[i]
            if i in [1, 4]:
                rg_f = rg_f + "."
            elif i == 7:
                rg_f = rg_f + "-"
        return rg_f

    def verf_rg(n_rg):
        dig_v = str(random.randint(0, 9))
        int_values = []
        peso = 2
        for i in range(len(n_rg)):
            int_values.append(int(n_rg[i]) * (peso + i))
        soma = sum(int_values)
        for x in range(10):
            result = soma + x * 100
            if result % 11 == 0:
                dig_v = str(x)
                break
        return dig_v

    seq_rg = ""
    for _ in range(8):
        random.seed()
        sel_num = random.randint(0, 9)
        seq_rg = seq_rg + str(sel_num)
    dig = verf_rg(seq_rg)
    seq_rg = seq_rg + dig
    seq_rg = formata_rg(seq_rg)
    return seq_rg

def make_name(tipo):
    # file = open("./files/nome.txt", "r", encoding="ISO-8859-1")
    # names = file.readlines()
    if tipo == "nome":
        nn = random.randint(2,4)
        if nn == 4:
            nn = 2
    else:
        nn = random.randint(2,3)

    full_name = ""
    for _ in range(nn):
        random.seed()
        sel_num = random.randint(0, n_names-1)
        full_name = full_name + name_dct[sel_num]
    full_name = full_name[:-1]
    return full_name

def make_surname():
    sel = random.randint(0, n_surnames-1)
    return surname_dct[sel].upper()

def get_nnomes(st):
    return len(st.split(' '))

def modelos_cnas():
    sel = random.randint(0, 2)
    doc = ""

    random.seed()
    sel_num = random.randint(0, len(df) - 1)
    sel_cid = df["Município"][sel_num].upper()
    sel_est = df["UF"].values[sel_num].upper()

    if sel == 0:
        c_nasc = 'CERT.NAS='
        a_folha = 'FL='
        livro = 'LV='

        lista = random.sample(range(1, 500), 3)
        c_nasc = c_nasc+str(lista[0])+' '
        livro = livro + str(lista[1])+' '
        a_folha = a_folha+str(lista[2])

        folha = c_nasc+livro+a_folha

        cart = random.randint(0, 5)
        cartorio = ["SEDE", '1a ZONA', '2a ZONA', '3 ZONA', '4 ZONA', '5 ZONA'][cart]

        doc = "CART. " + cartorio + "-" + sel_cid + " " + sel_est

    elif sel == 1:
        c_nasc = "CERT. NASCIMENTO CARTÓRIO: "
        termo = " TERMO:"
        folha = " FOLHA:"

        cart = random.randint(0, 5)
        cartorio = ["SEDE", '1a ZONA', '2a ZONA', '3 ZONA', '4 ZONA', '5 ZONA'][cart]

        term = random.randint(1, 1000000)
        termo += f"{term:07d}"

        fl = random.randint(1, 8000)
        folha += f"{fl:08d}"

        folha = c_nasc + cartorio + termo + folha

        liv = random.randint(0, 4)
        liv_num = random.randint(1, 10000)
        livro = ["a", "b", "c", "d", "e"][liv]
        livro = f"{liv_num:05d}" + livro

        doc = "LIVRO: " + livro + " " + sel_cid + " - " + sel_est

    if sel == 2:
        cn = " CN:LV."

        liv = random.randint(0, 4)
        liv_num = random.randint(1, 1000)
        livro = ["a", "b", "c", "d", "e"][liv]
        cn += livro + f"{liv_num:04d}"

        fl_num = random.randint(1, 1000)
        fl = f"FLS.{fl_num:04d}v"

        fl_num = random.randint(1, 195000)
        ns = f"N.{fl_num:04d}"

        folha = sel_cid + "-" + sel_est + cn + "/" + fl + "/" + ns

        d1 = "".join(random.choices(string.digits, k=6))
        d2 = "".join(random.choices(string.digits, k=2))
        d3 = "".join(random.choices(string.digits, k=2))
        d4 = "".join(random.choices(string.digits, k=4))
        d5 = "".join(random.choices(string.digits, k=1))
        d6 = "".join(random.choices(string.digits, k=5))
        d7 = "".join(random.choices(string.digits, k=3))
        d8 = "".join(random.choices(string.digits, k=7))
        d9 = "".join(random.choices(string.digits, k=2))
        
        doc = "MATRÍCULA: " + d1 + " " + d2 + " " + d3 + " " + d4 + \
                        " " + d5 + " " + d6 + " " + d7 + " " + d8 + " " + d9


    return [folha, doc]


class Person:
    def __init__(self):

        self.entities = {}

        self.set_filiacao(1)
        self.set_filiacao(2)
        self.set_nome()
        self.set_cpf()
        self.set_rg()
        self.set_cnh()

        self.set_pis()
        self.set_dni()
        self.set_cid_est(50)
        self.set_est()
        self.set_cid(50)

        self.set_d_orig()
        self.cnt = 0

        self.set_datanasc()
        self.set_dataexp()
        self.set_org()
        self.set_obs()
        self.set_cns()
        self.set_fator_rh()
        self.set_titulo()
        self.set_militar()

        self.set_profissional()
        self.cnt_p = 0

        self.set_ctps()
        self.set_serie()

    def get_entity(self, tipo):
        if tipo in self.entities:
            if tipo in ('datanasc', 'dataexp'):
                return time_to_str(self.entities[tipo])
            elif tipo == 'regcivil':
                if (self.cnt == 2):
                    return ""
                ret = self.entities[tipo][self.cnt]
                self.cnt += 1
                return ret
            elif tipo in ('filiacao1', 'filiacao2'):
                return self.entities[tipo]["name"]
            elif tipo == 'profissional':
                if self.cnt > 0:
                    return ""
                self.cnt += 1
                return self.entities[tipo]
            elif tipo == 'dni':
                return ""
            return self.entities[tipo]
        else:
            return None

    def set_nome(self):
        self.entities['nome'] = make_name("nome")
        if get_nnomes(self.entities['nome']) == 3:
            chosen = random.randint(1, 2)
            self.entities['nome'] += " " + self.entities[f"filiacao{chosen}"]["surname"]
        else:
            chosen = random.randint(0, 3)
            if chosen == 0:
                chosen = random.randint(1, 2)
                self.entities['nome'] += " " + self.entities[f"filiacao{chosen}"]["surname"]
            else:
                self.entities['nome'] += " " + self.entities[f"filiacao1"]["surname"]
                self.entities['nome'] += " " + self.entities[f"filiacao2"]["surname"]

        return self.entities['nome']

    def set_filiacao(self, num=1):
        full_name = make_name("filiacao")
        surname = make_surname()
        if num == 1:
            self.entities['filiacao1'] = {"name": full_name + " " + surname, "surname": surname}
        else:
            self.entities['filiacao2'] = {"name": full_name + " " + surname, "surname": surname}
        return full_name

    def set_cpf(self):
        def formata_cpf(n_cpf):
            formatado = ""
            for i in range(len(n_cpf)):
                formatado = formatado + n_cpf[i]
                if i in [2, 5]:
                    formatado = formatado + "."
                elif i == 8:
                    formatado = formatado + "-"
            return formatado

        def dig_verificador(n_cpf):
            int_values = []
            peso = 10 if len(n_cpf) == 9 else 11
            for i in range(len(n_cpf)):
                int_values.append(int(n_cpf[i]) * (peso - i))
            soma = sum(int_values)
            resto = soma % 11
            dig = "0" if resto in [0, 1] else str(11 - resto)
            return dig

        def make_cpf():
            seq_cpf = ""
            for _ in range(9):
                random.seed()
                sel_num = random.randint(0, 9)
                seq_cpf = seq_cpf + str(sel_num)
            f_dig = dig_verificador(seq_cpf)
            seq_cpf = seq_cpf + f_dig
            s_dig = dig_verificador(seq_cpf)
            seq_cpf = seq_cpf + s_dig
            seq_cpf = formata_cpf(seq_cpf)
            return seq_cpf

        r_cpf = make_cpf()
        self.entities['cpf'] = r_cpf
        return r_cpf

    def set_rg(self):
        r_rg = make_rg()
        self.entities['rg'] = r_rg
        return r_rg

    def set_pis(self):
        pis_pasep = ""
        for x in range(11):
            random.seed()
            sel_num = random.randint(0, 9)
            pis_pasep = pis_pasep + str(sel_num)
            if x == 3 or x == 8:
                pis_pasep = pis_pasep + "."
            elif x == 9:
                pis_pasep = pis_pasep + "-"
        # if len(pis_pasep) > qtd_chars:
        #     pis_pasep = pis_pasep[:qtd_chars]
        self.entities['pis'] = pis_pasep
        return pis_pasep

    def set_dni(self):
        dni = "*****"
        self.entities['dni'] = dni
        return dni

    def set_est(self):
        random.seed()
        df = pd.read_csv(r"./files/cid_est.csv", encoding="utf-8")
        sel_num = random.randint(0, df.shape[0] - 1)

        sel_est = df["UF"][sel_num].upper()
        self.entities['uf'] = sel_est
        return sel_est

    def set_cid(self, qtd_chars):
        df = pd.read_csv(r"./files/cid_est.csv", encoding="utf-8")
        df_filter = (
            df.loc[df["Município"].apply(lambda x: len(x) < qtd_chars)]
        ).dropna(how="all")

        if df_filter.shape[0] > 0:
            random.seed()
            sel_num = random.randint(0, df_filter.shape[0] - 1)
            sel_cid = df_filter["Município"].values[sel_num].upper()
        else:
            sel_cid = "ITU"

        self.entities['cid'] = sel_cid
        return sel_cid

    def set_cid_est(self, qtd_chars):
        df = pd.read_csv(r"./files/cid_est.csv", encoding="utf-8")
        df_filter = (
            df.loc[df["Município"].apply(lambda x: len(x) < qtd_chars - 4)]
        ).dropna(how="all")

        if df_filter.shape[0] > 0:
            random.seed()
            sel_num = random.randint(0, df_filter.shape[0] - 1)
            sel_est = df_filter["UF"].values[sel_num].upper()
            sel_cid = df_filter["Município"].values[sel_num].upper()
            sel_local = sel_cid + "-" + sel_est
        else:
            sel_local = "ITU-SP"

        self.entities['naturalidade'] = sel_local
        return sel_local

    # TODO: Checar get_cid
    def set_d_orig(self):
        res = modelos_cnas()

        self.entities['regcivil'] = [res[0], res[1]]
        return res

    def set_datanasc(self):
        if 'dataexp' not in self.entities:
            unf_date = gen_rand_datetime()
        else:
            unf_date = gen_rand_datetime(min_time=self.entities['dataexp'])
        self.entities['datanasc'] = unf_date
        return time_to_str(unf_date)

    def set_dataexp(self):
        if 'datanasc' not in self.entities:
            unf_date = gen_rand_datetime()
        else:
            unf_date = gen_rand_datetime(max_time=self.entities['datanasc'])
        self.entities['dataexp'] = unf_date
        return time_to_str(unf_date)

    def set_org(self):
        random.seed()
        sel_num = random.randint(0, len(org) - 1)
        text = org[sel_num]
        self.entities['orgaoexp'] = text
        return text

    def set_obs(self):
        text = ""
        random.seed()
        sel_num = random.randint(0, len(obs) - 1)
        text = obs[sel_num]
        self.entities['obs'] = text
        return text
    
    def set_fator_rh(self):
        rn = random.randint(0, len(fator_rh.keys())-1)
        rh = fator_rh[rn]
        self.entities['rh'] = rh
        return rh

    def set_titulo(self):
        # 1 out of 2
        if 'datanasc' not in self.entities or \
                    ((datetime.now() - self.entities['datanasc']).days//365) < 16 \
                             or random.randint(0, 1) == 0:
            titulo = "*****"
        else:
            titulo = "0"
            for i in range(0, 12):
                titulo += str(random.randint(0, 9))
        self.entities['te'] = titulo
        return titulo

    def set_ctps(self):
        # 1 out of 4
        if 'datanasc' not in self.entities or \
                ((datetime.now() - self.entities['datanasc']).days//365) < 14 \
                    or random.randint(0, 3) == 0 or 'cpf' not in self.entities:
            ctps = "*****"
        else:
            cpf = self.entities['cpf']
            ctps = cpf[0:3] + cpf[4:7] + cpf[8]
        self.entities['ctps'] = ctps
        return ctps

    def set_serie(self):
        if not 'ctps' not in self.entities or 'cpf' not in self.entities:
            serie = "*****"
        else:
            cpf = self.entities['cpf']
            serie = cpf[-5:-3] + cpf[-2:]
        self.entities['serie'] = serie
        return serie

    def set_cns(self):
        if random.randint(0, 2) != 0: # 2 out of 3
            cns = "*****"
        else:
            cns = ""
            for i in range(0, 15):
                cns += str(random.randint(0, 9))
        self.entities['cns'] = cns
        return cns

    def set_profissional(self):
        if random.randint(0, 2) != 0: # 2 out of 3
            profissional = "*****"
        else:
            if random.randint(0, 1) == 1:
                preamb = "CREA"
            else:
                preamb = "CONFEA"
            preamb = preamb + "/" + ufs[random.randint(0,len(ufs)-1)]

            profissional = preamb + " 0"
            for i in range(0, 7):
                profissional += str(random.randint(0, 9))
                if i in (0, 3):
                    profissional += "."
        self.entities['profissional'] = profissional 
        return profissional

    def set_militar(self):
        if random.randint(0, 5) != 0: # 2 out of 3
            militar = "*****"
        else:
            militar = ""
            for i in range(0, 7):
                militar += str(random.randint(0, 9))
        self.entities['militar'] = militar
        return militar

    def set_cnh(self):
        cnh = make_rg()
        self.entities['cnh'] = cnh
        return cnh
