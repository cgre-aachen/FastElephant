# ---------------------------------- Imports ----------------------------------
import numpy as np
import netCDF4
import os
#------------------------------------------------------------------------------
class MOOSEFiles:
    """This class contains all methods for manipulating the MOOSE related files."""

    def modify_input_file(self, file_name, parameters, additional_tag=""):
        """Performces a string replace for n parameters, where n is equal to the
           list length.

        Args:
        file_name = name (and path) of the MOOSE input file without extension
                    that should be modified
        parameters = parameters that needs to be inserted into the MOOSE
                     inputfile
        additional_tag = tag option for the name of the produced MOOSE
                         inputfile

        Returns:
        No return defined, the produced file is automatically saved in the
        folder of the provided MOOSE input file

        """

        inputfile = open(file_name + ".template", "r")
        lines = inputfile.read()

        for i in range (0, len(parameters)):
            place_holder = "$" + str(i) + "$"
            lines = str.replace(lines, place_holder, parameters[i])

        auto_inputfile = open(file_name + additional_tag + ".i", "w")
        auto_inputfile.write(lines)


    def read_vector_postprocessor_file(self, file_name, file_extension ,
                                       delimiter = " ",
                                       skip_header = 1, usecols = None):
        """Reads a vector postprocessor file.

        Args:
        file_name = name (and path) of the state vector
        file_extension = name of the file extension
        delimiter = delimiter used in this file (default value: " ")

        Returns:
        vector = state vector
        """
        vector = np.genfromtxt(file_name + file_extension,
                               delimiter = delimiter ,
                               skip_header=skip_header, usecols = usecols)
        return vector


    def extract_state_vector_from_exodus_file(self, file_name, variable_name='None'):
        """Reads a state vector from an exodus file.

        Args:
        file_name = name (and path) of the state vector

        Returns:
        vector = state vector
        """
        if(variable_name=="None"):
            variable_name = 'vals_nod_var1'
        state_file = netCDF4.Dataset(file_name + ".e")
        state_vector = state_file.variables[variable_name]

        return state_vector[:,:]


    def write_state_vector_to_exodus_file(self, file_name, state_vector):
        """Writes a state vector to an exodus file.

        Args:
        file_name = name (and path) of the state vector
        state_vector = state vector

        """
        state_file = netCDF4.Dataset(file_name + ".e", "r+")
        state_file.variables['vals_nod_var1'][0] = state_vector
        state_file.close()


    def write_reduced_state_vector_to_xdr_file(self, file_name, state_vector):
        """Transfers reduced to full state vector and writes it to an exodus file.

        Args:
        file_name = name (and path) of the reduced state vector
        state_vector = reduced state vector

        """
        f = open(file_name,"wb")
        p = xdrlib.Packer()
        p.pack_farray(online_N, state_vector, p.pack_double)
        f.write(p.get_buffer())
        f.close()


    def read_obs_data_points(self,file_name, usecols ,skip_header = 1,
                             skip_footer = 0,
                             delimiter = " "):
        """Reads a postprocessor file.

        Args:
        file_name = name (and path) of the observation points
        usecols = columns that are used during the file reading process
        skip_header = defines how many lines are skipped at the beginning of
                      the file
                      (default value: 1)
        skip_footer = defines how many lines are skipped at the end of the file
                      (default value: 0)

        Returns:
        obs_points[n_obs] = observation points
        """

        obs_points = np.genfromtxt(file_name + ".csv", delimiter = delimiter,
                                   skip_header=skip_header,
                                   skip_footer = skip_footer,
                                   usecols = usecols)
        return obs_points


    def perform_offline_stage(self, num_timesteps, file_offline_stage, path_executable):
        """Performs the offline stage.
        Args:
            num_timesteps = number of timesteps
            file_offline_stage = name (and path) to the inputfile for the offline stage

        Returns:
            no return arguments
        """

        self.modify_input_file(file_offline_stage, num_timesteps)

        os.system(path_executable + " -i " + file_offline_stage + ".i")
